import warnings
from typing import Dict, Iterable, List

import aesara.tensor as at
from aesara.graph.basic import Variable, ancestors
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.tensor.rewriting.shape import ShapeFeature
from aesara.tensor.var import TensorVariable


def simplify_shapes(graphs: List[Variable]):
    """Simply the shape calculations in a list of graphs."""
    shape_fg = FunctionGraph(
        outputs=graphs,
        features=[ShapeFeature()],
        clone=False,
    )
    return rewrite_graph(shape_fg).outputs


class RaveledParamsMap:
    """Maps a set of tensor variables to a vector of their raveled values."""

    def __init__(self, ref_params: Iterable[TensorVariable]):
        self.ref_params = tuple(ref_params)

        self.ref_shapes = [at.shape(p) for p in self.ref_params]
        self.ref_shapes = simplify_shapes(self.ref_shapes)

        self.ref_dtypes = [p.dtype for p in self.ref_params]

        ref_shapes_ancestors = set(ancestors(self.ref_shapes))
        uninferred_shape_params = [
            p for p in self.ref_params if (p in ref_shapes_ancestors)
        ]
        if any(uninferred_shape_params):
            # After running the shape optimizations, the graphs in
            # `ref_shapes` should not depend on `ref_params` directly.
            # If they do, it could imply that we need to sample parts of a
            # model in order to get the shapes/sizes of its parameters, and
            # that's a worst-case scenario.
            warnings.warn(
                "The following parameters need to be computed in order to determine "
                f"the shapes in this parameter map: {uninferred_shape_params}"
            )

        param_sizes = [at.prod(s) for s in self.ref_shapes]
        cumsum_sizes = at.cumsum(param_sizes)
        # `at.cumsum` doesn't return a tensor of a fixed/known size
        cumsum_sizes = [cumsum_sizes[i] for i in range(len(param_sizes))]
        self.slice_indices = list(zip([0] + cumsum_sizes[:-1], cumsum_sizes))
        self.vec_slices = [slice(*idx) for idx in self.slice_indices]

    def ravel_params(self, params: List[TensorVariable]) -> TensorVariable:
        """Concatenate the raveled vectors of each parameter."""
        return at.concatenate([at.atleast_1d(p).ravel() for p in params])

    def unravel_params(
        self, raveled_params: TensorVariable
    ) -> Dict[TensorVariable, TensorVariable]:
        """Unravel a concatenated set of raveled parameters."""
        return {
            k: v.reshape(s).astype(t)
            for k, v, s, t in zip(
                self.ref_params,
                [raveled_params[slc] for slc in self.vec_slices],
                self.ref_shapes,
                self.ref_dtypes,
            )
        }

    def __repr__(self):
        return f"{type(self).__name__}({self.ref_params})"
