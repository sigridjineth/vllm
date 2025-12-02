# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional

import msgspec

if TYPE_CHECKING:
    from vllm.config import ModelConfig


class PoolingParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """API parameters for pooling models.

    Attributes:
        dimensions: Reduce the dimensions of embeddings
                    if model support matryoshka representation.
        additional_data: Any additional data needed for pooling.
        is_document: Whether this is a document embedding (for FDE pooler).
        normalize: Whether to L2 normalize the output embeddings.
    """

    dimensions: Optional[int] = None
    additional_data: Optional[Any] = None
    is_document: bool = False
    normalize: bool = False

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return PoolingParams(dimensions=self.dimensions,
                             additional_data=self.additional_data,
                             is_document=self.is_document,
                             normalize=self.normalize)

    def verify(self, model_config: "ModelConfig") -> None:
        if self.dimensions is not None:
            if not model_config.is_matryoshka:
                raise ValueError(
                    f'Model "{model_config.served_model_name}" does not '
                    f'support matryoshka representation, '
                    f'changing output dimensions will lead to poor results.')

            mds = model_config.matryoshka_dimensions
            if mds is not None:
                if self.dimensions not in mds:
                    raise ValueError(
                        f'Model "{model_config.served_model_name}" '
                        f'only supports {str(mds)} matryoshka dimensions, '
                        f'use other output dimensions will '
                        f'lead to poor results.')
            elif self.dimensions < 1:
                raise ValueError("Dimensions must be greater than 0")

    def __repr__(self) -> str:
        return (f"PoolingParams("
                f"dimensions={self.dimensions}, "
                f"additional_metadata={self.additional_data})")
