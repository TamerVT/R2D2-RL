import mujoco


def random_position_upper_right(pos, minx, maxx, miny, maxy, z, random):
    del pos
    return (
        random.uniform((minx + maxx) / 2, maxx),
        random.uniform((miny + maxy) / 2, maxy),
        z,
    )


def random_position_around_pos(pos, minx, maxx, miny, maxy, z, random):
    return (
        pos[0] + random.uniform(-1, 1) * (maxx - minx),
        pos[1] + random.uniform(-1, 1) * (maxy - miny),
        pos[2] + z,
    )


def random_position_in_bounds(pos, minx, maxx, miny, maxy, z, random):
    return (
        max(minx, min(maxx, pos[0] + random.uniform(-0.3, 0.3) * (maxx - minx))),
        max(miny, min(maxy, pos[1] + random.uniform(-0.3, 0.3) * (maxy - miny))),
        z,
    )


def get_geom_pos(model, data, name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id == -1:
        msg = f"Geom '{name}' not found"
        raise ValueError(msg)
    return data.geom_xpos[geom_id]
