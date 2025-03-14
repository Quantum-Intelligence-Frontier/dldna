import os
import textwrap

def create_project_structure():
    # 프로젝트 기본 구조 생성
    project_name = "simple_mistral"
    directories = [
        project_name,
        f"{project_name}/utils",
        f"{project_name}/tests",
    ]
    
    # 디렉토리 생성
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "__init__.py"), "w") as f:
            f.write("")

    # 파일 내용 정의
    kv_cache_content = textwrap.dedent("""
        import torch
        import torch.nn as nn
        
        class KVCacheManager:
            def __init__(self, config):
                self.config = config
    """)

    attention_content = textwrap.dedent("""
        import torch
        import torch.nn as nn
        from .utils.rotary_embedding import RotaryEmbedding
        from .kv_cache import KVCacheManager
    """)

    config_content = textwrap.dedent("""
        class MistralConfig:
            def __init__(self):
                self.hidden_size = 4096
                self.num_attention_heads = 32
                self.num_key_value_heads = 8
    """)

    rotary_content = textwrap.dedent("""
        import torch
        import torch.nn as nn
        
        class RotaryEmbedding(nn.Module):
            def __init__(self, dim, max_position_embeddings=2048):
                super().__init__()
                self.dim = dim
                self.max_position_embeddings = max_position_embeddings
    """)

    # 파일 생성
    files = {
        f"{project_name}/attention.py": attention_content,
        f"{project_name}/config.py": config_content,
        f"{project_name}/kv_cache.py": kv_cache_content,
        f"{project_name}/utils/rotary_embedding.py": rotary_content,
    }

    for file_path, content in files.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Project {project_name} has been created successfully!")

if __name__ == "__main__":
    create_project_structure()