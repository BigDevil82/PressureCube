def generate_steel_frame(L, W, s, d):
    # 生成框架梁坐标
    frame_x = {0, W}
    frame_y = {0, L}

    # 生成主梁坐标（纵向，Y轴方向）
    primary_y = set()
    y = s
    while y < W:
        if y not in frame_y:
            primary_y.add(y)
        y += s

    # 生成次梁坐标（横向，X轴方向）
    secondary_x = set()
    x_mid = W / 2
    secondary_x.add(x_mid)

    # 向右扩展
    x = x_mid + s
    while x < W:
        secondary_x.add(x)
        x += s

    # 向左扩展
    x = x_mid - s
    while x > 0:
        secondary_x.add(x)
        x -= s

    # 生成支撑梁坐标（在次梁之间）
    support_x = set()
    sorted_secondary = sorted(secondary_x)
    for i in range(len(sorted_secondary) - 1):
        x_start = sorted_secondary[i]
        x_end = sorted_secondary[i + 1]

        x_current = x_start + d
        while x_current < x_end:
            support_x.add(x_current)
            x_current += d

    # 合并所有垂直坐标（X轴）
    vertical_x = frame_x.union(secondary_x).union(support_x)
    vertical_x = sorted(vertical_x)

    # 合并所有水平坐标（Y轴）
    horizontal_y = frame_y.union(primary_y)
    horizontal_y = sorted(horizontal_y)

    # 生成所有线段
    segments = []

    # 处理竖直梁（框架梁和主梁）
    for y in horizontal_y:
        beam_type = "frame" if y in frame_y else "primary"
        for i in range(len(vertical_x) - 1):
            x1 = vertical_x[i]
            x2 = vertical_x[i + 1]
            segments.append(((x1, y), (x2, y), beam_type))

    # 处理水平梁（框架梁、次梁、支撑梁）
    for x in vertical_x:
        if x in frame_x:
            beam_type = "frame"
        elif x in secondary_x:
            beam_type = "secondary"
        else:
            beam_type = "support"

        for i in range(len(horizontal_y) - 1):
            y1 = horizontal_y[i]
            y2 = horizontal_y[i + 1]
            segments.append(((x, y1), (x, y2), beam_type))

    return segments
