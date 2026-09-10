"""ONNX Runtime embedding pipeline with CUDA acceleration and CPU fallback."""

from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np


class Embedder:
    """Generates dense vector embeddings using ONNX Runtime with CUDA / CPU fallback."""

    def __init__(
        self,
        session: Any = None,
        tokenizer: Any = None,
        dim: int = 384,
        provider: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 64,
    ):
        self.session = session
        self.tokenizer = tokenizer
        self.dim = dim
        self._provider = provider
        self.max_length = max_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(42)

        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "enable_padding"):
                self.tokenizer.enable_padding()
            if hasattr(self.tokenizer, "enable_truncation"):
                self.tokenizer.enable_truncation(max_length=self.max_length)

    @property
    def is_mock(self) -> bool:
        """True if running in mock/deterministic simulation mode without active ONNX session."""
        return self.session is None or self.tokenizer is None

    @property
    def provider(self) -> str:
        """Active execution provider name ('CUDAExecutionProvider', 'CPUExecutionProvider', or 'mock')."""
        if self.is_mock:
            return "mock"
        if self._provider is not None:
            return self._provider
        if hasattr(self.session, "get_providers"):
            providers = self.session.get_providers()
            return providers[0] if providers else "unknown"
        return "onnxruntime"

    @classmethod
    def get_preferred_providers(cls) -> List[str]:
        """Return available ONNX Runtime execution providers in priority order (CUDA, then CPU)."""
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            preferred: List[str] = []
            if "CUDAExecutionProvider" in available:
                preferred.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                preferred.append("CPUExecutionProvider")
            return preferred if preferred else available
        except ImportError:
            return ["CPUExecutionProvider"]

    @classmethod
    def create_mock_or_real(
        cls,
        use_mock: bool = False,
        model_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        dim: int = 384,
    ) -> "Embedder":
        """Factory method to initialize Embedder with CUDA/CPU provider or fallback to mock mode."""
        if use_mock:
            return cls(session=None, tokenizer=None, dim=dim)

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            # Check explicit paths
            if model_path is not None and tokenizer_path is not None:
                m_path = Path(model_path)
                t_path = Path(tokenizer_path)
                if m_path.exists() and t_path.exists():
                    providers = cls.get_preferred_providers()
                    session = ort.InferenceSession(str(m_path), providers=providers)
                    tokenizer = Tokenizer.from_file(str(t_path))
                    return cls(session=session, tokenizer=tokenizer, dim=dim)

            # Check default cache directory
            try:
                from hyperindex.config import get_config

                config = get_config()
                cached_model = config.cache_dir / "models" / "model.onnx"
                alt_cached_model = config.cache_dir / "models" / "onnx" / "model.onnx"
                cached_tok = config.cache_dir / "models" / "tokenizer.json"

                actual_model = (
                    cached_model
                    if cached_model.exists()
                    else (alt_cached_model if alt_cached_model.exists() else None)
                )
                if actual_model is not None and cached_tok.exists():
                    providers = cls.get_preferred_providers()
                    session = ort.InferenceSession(str(actual_model), providers=providers)
                    tokenizer = Tokenizer.from_file(str(cached_tok))
                    return cls(session=session, tokenizer=tokenizer, dim=dim)
            except Exception:
                pass

            # Fallback wrapper if model not yet downloaded
            return cls(session=None, tokenizer=None, dim=dim)
        except ImportError:
            return cls(session=None, tokenizer=None, dim=dim)

    @classmethod
    def download_model(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        target_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Download ONNX model and tokenizer from Hugging Face Hub to target directory."""
        from huggingface_hub import hf_hub_download

        if target_dir is None:
            from hyperindex.config import get_config

            target_dir = get_config().cache_dir / "models"
        dest_dir = Path(target_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id=model_name,
            filename="onnx/model.onnx",
            local_dir=str(dest_dir),
        )
        hf_hub_download(
            repo_id=model_name,
            filename="tokenizer.json",
            local_dir=str(dest_dir),
        )
        return dest_dir

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Run tokenization, inference, pooling, and normalization on a single batch."""
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if hasattr(self.session, "get_inputs"):
            session_inputs = {inp.name for inp in self.session.get_inputs()}
            if "token_type_ids" in session_inputs:
                feed["token_type_ids"] = np.array(
                    [getattr(e, "type_ids", [0] * len(e.ids)) for e in encoded],
                    dtype=np.int64,
                )

        outputs = self.session.run(None, feed)

        # Mean pooling + unit normalization
        token_embeddings = outputs[0]
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate normalized embeddings for a list of text strings.

        Returns an array of shape (len(texts), dim) with float32 values normalized to unit length.
        Batches inference to avoid out-of-memory errors on large inputs.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        chunk_size = batch_size if batch_size is not None else self.batch_size

        if self.session is None or self.tokenizer is None:
            # Deterministic pseudo-embedding for testing/mock mode using isolated RNG
            raw = self.rng.standard_normal((len(texts), self.dim)).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            return (raw / np.maximum(norms, 1e-12)).astype(np.float32)

        # Real ONNX Runtime inference in batches
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), chunk_size):
            batch_texts = texts[i : i + chunk_size]
            all_embeddings.append(self._embed_batch(batch_texts))

        if len(all_embeddings) == 1:
            return all_embeddings[0]
        return np.vstack(all_embeddings)
