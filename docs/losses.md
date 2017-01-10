[losses.py](https://github.com/codekansas/gandlf/blob/master/gandlf/losses.py)

## Negative Binary Crossentropy

Instead of minimizing `log(1-D)` maximize `log(D)`.

Note that when using this loss function, you should not change the target. For example, if you want `G -> 0` and `D -> 1` to train your generator, you should replace your `binary_crossentropy` for the fake output with `negative_binary_crossentropy` while keeping the target output as 0.

## Maximize

Maximizes `y_true`, regardless of `y_pred`.

## Minimize

Minimizes `y_true`, regardless of `y_pred`.

