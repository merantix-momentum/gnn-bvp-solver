from typing import Any, Callable, Iterable, Tuple, Union
import fenics as fn


class RecursiveUserExpression(fn.UserExpression):
    def __init__(self, default: int = 0, **kwargs):
        """Recursive fenics user expression. Defines a function that can be evaluated
        and states physical quantities at all locations.

        Args:
            default (int, optional): default value for this expression. Defaults to 0.
        """
        self.default = default
        self.sub_expressions = []
        super().__init__(degree=1, **kwargs)

    def add_subexpression(self, e: Callable[[float, float], Union[float, None]]) -> None:
        """Add a subexpression that can change a local part of the quantity.

        Args:
            e (Callable[[float, float], Union[float, None]]): Callable expression.

        Returns:
            None
        """
        self.sub_expressions.append(e)

    def eval_cell(self, values: Any, point: Iterable[float], cell: Any) -> float:
        """Override base method to tell fenics which values the quantity has at which point.

        Args:
            values: value array to store values into
            point (Iterable[float]): coordinates of the current point where we evaluate.
            cell: not used

        Returns:
            float: value of physical quantity
        """
        values[0] = self(point)
        return values[0]

    def __call__(self, point: Iterable[float], *args) -> float:
        """Expression is callable and outputs a value for each point.

        Args:
            point (Iterable[float]): coordinates of the current point.

        Returns:
            float: value of physical quantity
        """
        result = self.default
        x, y = point[0], point[1]

        # check all candidates for that node
        # the last one wins if there is a conflict
        for e in self.sub_expressions:
            candidate = e(x, y, *args)
            if candidate is not None:
                result = candidate

        return result

    def value_shape(self) -> Tuple:
        """Shape of the ph. quantity

        Returns:
            Tuple: empty tuple to indicate scalar quantity
        """
        return ()
