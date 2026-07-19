"""
基于SQLite的内存存储模块
提供langgraph的SQLite存储接口,即AsyncSqliteSaver，用于保存agent的检查点状态

【批量写入优化】
引入 BatchedAsyncSqliteSaver 包装器，将多次 aput 调用批量写入 SQLite，
减少磁盘 I/O 频率，在高频对话场景中显著降低写入开销。

写入策略：
- 积攒 BATCH_SIZE（默认=5）个检查点后一次性写入
- 在读取检查点（aget_tuple）前自动刷入，保证状态一致性
- 可在会话结束时显式调用 flush() 确保数据持久化
"""

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
import os
from typing import Optional, Any, AsyncIterator

# ══════════════════════════════════════════════════════════════
# 导入第三方库
# ══════════════════════════════════════════════════════════════
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, ChannelVersions,CheckpointTuple
from langgraph.types import RunnableConfig

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════
from core.config.settings import Config
from .logger import LoggerManager

logger = LoggerManager.get_logger()

# ══════════════════════════════════════════════════════════════
# 默认常量
# ══════════════════════════════════════════════════════════════

# ── 默认批量大小 ──
# 每个 aput 在内部产生一条 INSERT 语句。
# 积攒 BATCH_SIZE 个后一次性提交事务，大幅减少 fsync 次数。
BATCH_SIZE = 5


# ══════════════════════════════════════════════════════════════
# 检查点批量写入包装器
# ══════════════════════════════════════════════════════════════

class BatchedAsyncSqliteSaver(AsyncSqliteSaver):
    """
    批量写入的 AsyncSqliteSaver（继承 AsyncSqliteSaver，满足 LangGraph 类型检查）。

    将 aput / aput_writes 调用缓冲在内存中，
    达到 BATCH_SIZE 或显式 flush() 时再批量写入 SQLite。
    读操作（aget_tuple / alist）会触发自动刷入，保证数据一致性。

    用法:
        inner = await get_async_sqlite_saver()
        batched = BatchedAsyncSqliteSaver(conn=inner.conn, batch_size=5)
        # 传入 create_agent(checkpointer=batched)
        # ...
        await batched.flush()  # 会话结束时确保写入
    """

    # ── 初始化 ──
    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        serde: Optional[Any] = None,
        batch_size: int = BATCH_SIZE,
    ):
        super().__init__(conn=conn, serde=serde)
        self._batch_size = batch_size    #出发批量写入的阈值
        # 缓冲区：列表元素为 (config, checkpoint, metadata, new_versions)
        self._buffer: list = []  # 存储 aput（保存 checkpoint）的待处理数据
        self._writes_buffer: list = []  # 存储 aput_writes（保存子任务）的待处理数据
        self._flushing = False  # 防止重入写入操作

    # ── 写操作：缓冲 ──

    #缓冲检查点写入
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """缓冲检查点写入"""
        self._buffer.append((config, checkpoint, metadata, new_versions))
        if len(self._buffer) >= self._batch_size and not self._flushing:
            await self._flush()
        # 返回 checkpoint 的 id（LangGraph 需要）
        return config

    #缓冲子任务写入
    # 缓冲区：aput_writes 的缓冲区
    #保存图执行过程中产生的中间写入操作（即节点对状态通道的写入记录）
    from typing import Sequence
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """缓冲子任务写入"""
        self._writes_buffer.append((config, writes, task_id, task_path))
        if len(self._writes_buffer) >= self._batch_size and not self._flushing:
            await self._flush_writes()

    # ── 读操作：先刷后读 ──

    #读取单个检查点
    async def aget_tuple(
        self, config: RunnableConfig
    ) -> Optional["CheckpointTuple"]:
        """读取检查点（自动刷入缓冲）"""
        #如果 _buffer 中有未写入的 checkpoint 数据，且当前不在刷写状态，
        #则调用 _flush() 将缓冲区中的所有 checkpoint 批量写入 SQLite
        if self._buffer and not self._flushing:
            await self._flush()
        #如果 _writes_buffer 中有未写入的 writes 数据
        #则调用 _flush_writes() 批量写入
        if self._writes_buffer and not self._flushing:
            await self._flush_writes()
        #调用父类（AsyncSqliteSaver）的 aget_tuple，从数据库读取实际数据
        return await super().aget_tuple(config)

    #列出检查点
    async def alist(
        self,
        config: Optional["RunnableConfig"] = None,
        *,
        filter: Optional[dict] = None,
        before: Optional["RunnableConfig"] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator["CheckpointTuple"]:
        """列出检查点（自动刷入缓冲）"""
        if self._buffer and not self._flushing:
            await self._flush()
        if self._writes_buffer and not self._flushing:
            await self._flush_writes()
        async for item in super().alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    # ── 刷入操作 ──

    #执行实际批量写入
    async def flush(self):
        """显式刷入所有缓冲的检查点。会话结束时调用。"""
        #如果已经有刷写操作在进行，直接返回（避免并发执行导致状态错乱）
        if self._flushing:
            return
        self._flushing = True
        #先刷写 checkpoint 缓冲区（_buffer），再刷写 writes 缓冲区（_writes_buffer）
        #确保检查点和子任务的写入顺序一致
        try:
            checkpoint_count = 0
            writes_count = 0
            if self._buffer:
                checkpoint_count = await self._flush()
            if self._writes_buffer:
                writes_count = await self._flush_writes()
            if checkpoint_count or writes_count:
                logger.debug(
                    f"[检查点批量] 缓冲已刷入: checkpoints={checkpoint_count}, writes={writes_count}"
                )
        finally:
            self._flushing = False

    async def _flush(self):
        """将缓冲的检查点批量写入 SQLite"""
        if not self._buffer:
            return 0
        batch = self._buffer[:]
        self._buffer = []
        for config, checkpoint, metadata, new_versions in batch:
            await super().aput(config, checkpoint, metadata, new_versions)
        logger.debug(f"[检查点批量] 写入 {len(batch)} 个检查点")
        return len(batch)

    async def _flush_writes(self):
        """将缓冲的子任务写入批量写入 SQLite"""
        if not self._writes_buffer:
            return 0
        batch = self._writes_buffer[:]
        self._writes_buffer = []
        for config, writes, task_id, task_path in batch:
            await super().aput_writes(config, writes, task_id, task_path)
        logger.debug(f"[检查点批量] 写入 {len(batch)} 个子任务")
        return len(batch)


# ══════════════════════════════════════════════════════════════
# 原始 saver 工厂
# ══════════════════════════════════════════════════════════════

# 确保数据库的路径存在
def ensure_db_dir(db_path: str):
    """
    确保数据库文件所在目录存在
    :param db_path: 数据库文件路径
    :return: None
    """
    # 从完整数据库路径中提取目录部分，若目录不存在则递归创建。
    dir_path = os.path.dirname(db_path)         #提取目录路径
    if not os.path.exists(dir_path):            #判断目录是否存在
        try:
            os.makedirs(dir_path, exist_ok=True)             #创建目录
            logger.info(f"创建数据库目录: {dir_path}")        #记录日志
        except Exception as e:
            logger.error(f"创建数据库目录失败: {dir_path}, 错误信息: {e}")
            raise e
        
# 初始化SQLlitesaver实例，单例的存储容器，初始为 None
_sqlite_saver_instance = None

# ── 获取原始 SqliteSaver 单例 ──
async def get_async_sqlite_saver() -> AsyncSqliteSaver:         #返回AsyncSqliteSaver实例
    """
    获取SqliteSaver实例，确保单例模式,链接配置的memory_db_path数据库
    :return: SqliteSaver实例
    """
    #如果实例尚未创建，进入初始化流程；否则直接返回已有实例。
    global _sqlite_saver_instance
    if _sqlite_saver_instance is None:
        try:
            ensure_db_dir(Config.MEMORY_DB_PATH)             #确保数据库目录存在
            db_path = Config.MEMORY_DB_PATH                  #用db_path参数创建一个AsyncSqliteSaver实例并赋值给_sqlite_saver_instance
            # 异步连接数据库
            conn = await aiosqlite.connect(db_path,check_same_thread=False)
            #连接数据库,允许该连接在其他线程中使用
            await conn.execute("PRAGMA journal_mode = WAL;")
            #设置数据库的journal模式为WAL（Write-Ahead Logging），提高并发性能和数据安全性
            _sqlite_saver_instance = AsyncSqliteSaver(conn=conn)
            if hasattr(_sqlite_saver_instance, "setup"):
                await _sqlite_saver_instance.setup()
            #使用已经配置好的 aiosqlite.Connection 对象来创建 AsyncSqliteSaver 实例
            logger.info(f"AsyncSqliteSaver实例创建成功，数据库路径: {db_path}")
        except Exception as e:
            logger.error(f"创建AsyncSQLiteSaver实例失败，错误信息: {e}")
            raise e
    return _sqlite_saver_instance


# ══════════════════════════════════════════════════════════════
# 批量写入 saver 工厂
# ══════════════════════════════════════════════════════════════

_batched_saver_instance = None

# ── 获取批量写入 Saver 单例 ──
async def get_batched_sqlite_saver(batch_size: int = BATCH_SIZE) -> BatchedAsyncSqliteSaver:
    """
    获取批量写入的检查点保存器。

    在 get_async_sqlite_saver() 的基础上包裹 BatchedAsyncSqliteSaver，
    减少高频对话场景的 SQLite I/O 开销。

    Args:
        batch_size: 批量大小，默认 5。积攒到该数量后一次性写入。

    Returns:
        BatchedAsyncSqliteSaver 实例

    用法:
        checkpointer = await get_batched_sqlite_saver()
        agent = create_agent(..., checkpointer=checkpointer)
        # ...
        await checkpointer.flush()  # 会话结束时
    """
    global _batched_saver_instance
    if _batched_saver_instance is not None:
        return _batched_saver_instance

    inner = await get_async_sqlite_saver()
    # 使用 inner 的数据库连接创建 BatchedAsyncSqliteSaver
    # 数据库连接会传给父类初始化，确保实例检查通过。
    _batched_saver_instance = BatchedAsyncSqliteSaver(
        conn=inner.conn,
        serde=inner.serde,
        batch_size=batch_size,
    )
    logger.info(f"BatchedAsyncSqliteSaver 创建完成 (batch_size={batch_size})")
    return _batched_saver_instance


# ══════════════════════════════════════════════════════════════
# 公共操作
# ══════════════════════════════════════════════════════════════

async def clear_memory():
    global _sqlite_saver_instance, _batched_saver_instance
    db_path = Config.MEMORY_DB_PATH
    # 如果有实例，先关闭连接
    seen = set()
    for inst in [_sqlite_saver_instance, _batched_saver_instance]:
        conn = getattr(inst, "conn", None) if inst is not None else None
        if conn is not None and id(conn) not in seen:
            seen.add(id(conn))
            await conn.close()
    _sqlite_saver_instance = None
    _batched_saver_instance = None
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.warning(f"内存数据库已清空，删除文件: {db_path}")
        except Exception as e:
            logger.error(f"清空内存数据库失败，错误信息: {e}")
            raise e
    else:
        logger.info(f"内存数据库文件不存在，无需清空: {db_path}")

async def close_sqlite_saver():
    global _sqlite_saver_instance, _batched_saver_instance
    seen = set()
    for inst in [_sqlite_saver_instance, _batched_saver_instance]:
        conn = getattr(inst, "conn", None) if inst is not None else None
        if conn is not None and id(conn) not in seen:
            seen.add(id(conn))
            await conn.close()
    _sqlite_saver_instance = None
    _batched_saver_instance = None
    logger.info(f"SQL数据连接已关闭.")


# ══════════════════════════════════════════════════════════════
# 测试入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("正在测试BatchedAsyncSqliteSaver...")
    import asyncio

    async def test():
        saver = None
        try:
            # 获取批量 saver
            batched = await get_batched_sqlite_saver(batch_size=3)
            print(f"BatchedAsyncSqliteSaver获取成功: {batched}")

            # 构造测试检查点
            test_config = {"configurable": {"thread_id": "test_thread","checkpoint_ns": "test_ns"}}
            test_checkpoint = {
                "v": 1,
                "ts": {},
                "id": "1234",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {}
            }
            test_metadata = {"source": "test", "step": 0}
            test_new_versions = {}

            # 写入多个检查点（应缓冲）
            for i in range(4):
                cp = await batched.aput(
                    config=test_config,
                    checkpoint={**test_checkpoint, "id": f"cp_{i}"},
                    metadata=test_metadata,
                    new_versions=test_new_versions,
                )
                print(f"  写入检查点 {i}，缓冲大小={len(batched._buffer)}")

            # 显式刷入
            await batched.flush()
            print(f"  刷入后缓冲区大小={len(batched._buffer)}")

        finally:
            await close_sqlite_saver()
            print("数据库连接已关闭")

    asyncio.run(test())
