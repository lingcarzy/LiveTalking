# configs/base_config.py
import argparse
import yaml
import os
from pydantic import BaseModel, Field
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any

# 定义配置数据类
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    listen_port: int = 8010
    transport: str = "webrtc"
    push_url: str = ""
    max_session: int = 1

class ModelConfig(BaseModel):
    name: str = Field("musetalk", pattern="^(musetalk|wav2lip|ultralight)$")
    device: str = "cuda"
    fps: int = Field(50, gt=0, description="FPS must be positive")
    batch_size: int = Field(8, gt=0)
    avatar_id: str

@dataclass
class TTSConfig:
    engine: str = "edgetts"
    server: str = ""
    ref_file: str = ""
    ref_text: Optional[str] = None

@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    system_msg: str = "You are a helpful assistant."

@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # 运行时参数，通常不在配置文件中写死
    customopt: List[Any] = field(default_factory=list)
    session_id: int = 0

    @classmethod
    def from_yaml(cls, path: str):
        """从 YAML 文件加载配置"""
        if not os.path.exists(path):
            logger.warning(f"Config file {path} not found, using defaults.")
            return cls()
            
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
        # 构建嵌套配置对象
        server_cfg = ServerConfig(**data.get('server', {}))
        model_cfg = ModelConfig(**data.get('model', {}))
        tts_cfg = TTSConfig(**data.get('tts', {}))
        llm_cfg = LLMConfig(**data.get('llm', {}))

        return cls(
            server=server_cfg,
            model=model_cfg,
            tts=tts_cfg,
            llm=llm_cfg
        )

    @classmethod
    def from_args(cls):
        """优先级：命令行参数 > 配置文件 > 默认值"""
        parser = argparse.ArgumentParser(description="LiveTalking Refactored Server")
        
        # 核心参数：配置文件路径
        parser.add_argument('--config', type=str, default="configs/default.yaml", 
                            help="Path to the YAML configuration file.")
        
        # 覆盖参数 (可选，用于快速测试或临时调整)
        parser.add_argument('--listenport', type=int, help="Override server port")
        parser.add_argument('--avatar_id', type=str, help="Override avatar ID")
        parser.add_argument('--model', type=str, choices=['musetalk', 'wav2lip', 'ultralight'], help="Override model name")
        
        args = parser.parse_args()

        # 1. 加载配置文件
        config = cls.from_yaml(args.config)

        # 2. 应用命令行覆盖
        if args.listenport:
            config.server.listen_port = args.listenport
        if args.avatar_id:
            config.model.avatar_id = args.avatar_id
        if args.model:
            config.model.name = args.model
            
        return config

    def __str__(self):
        return yaml.dump(asdict(self), default_flow_style=False, allow_unicode=True)

# 引入 logger (防止循环导入，这里简单处理)
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)