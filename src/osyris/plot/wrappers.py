import numpy as np


def quiver(ax, x, y, z, density=1, color="w", **kwargs):

    default_args = {
        "angles": "xy",
        "pivot": "mid",
        # "color": "w"
        }

    default_args.update(kwargs)

    skips = np.around(np.array(z.shape) * 4.0 / 128.0 / density).astype(np.int)
    # skipx = int(round(4.0 / density))
    # skipy = int(round(4.0 / density))
    skip = (slice(None,None,skips[0]),slice(None,None,skips[1]))

    args = [x[skip[0]], y[skip[1]], z[..., 0][skip], z[..., 1][skip]]
    if isinstance(color, str):
        default_args["color"] = color
    else:
        args.append(z[..., 2][skip])

    return ax.quiver(*args, **default_args)


def pcolormesh(ax, x, y, z, **kwargs):

    default_args = {
        "shading": "nearest",
        }

    default_args.update(kwargs)

    return ax.pcolormesh(x, y, z, **default_args)


def contour(ax, x, y, z, labels=True, **kwargs):

    cs = ax.contour(x, y, z, **kwargs)
    if labels:
        ax.clabel(cs, inline=1, fontsize=10)
    return cs

def contourf(ax, x, y, z, **kwargs):

    return ax.contourf(x, y, z, **kwargs)


def streamplot(ax, x, y, z, **kwargs):

    default_args = {
        "color": "w"
        }

    default_args.update(kwargs)

    return ax.streamplot(x, y, z[..., 0], z[..., 1], **default_args)
