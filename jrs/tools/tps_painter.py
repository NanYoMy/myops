


from PIL import Image, ImageDraw, ImageOps


def save_image_with_tps_points(source_points, img, outdir, name, grid_size=4, img_size=128, border=0):

    img = Image.fromarray(img).convert('RGB').resize((img_size, img_size))
    # source_points = (source_points + 1) / 2 * 128 + 64
    # source_image = Image.fromarray(img).convert('RGB').resize((128, 128))

    canvas = Image.new(mode='RGB', size=(img_size + 2 * border, img_size + 2 * border), color=(128, 128, 128))
    canvas.paste(img, (border, border))
    source_points = (source_points + 1) / 2 * img_size + border
    draw = ImageDraw.Draw(canvas)

    for x, y in source_points:
        draw.rectangle([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 0))

    source_points = source_points.view(grid_size, grid_size, 2)
    for j in range(grid_size):
        for k in range(grid_size):
            x1, y1 = source_points[j, k]
            if j > 0:  # connect to left
                x2, y2 = source_points[j - 1, k]
                draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
            if k > 0:  # connect to up
                x2, y2 = source_points[j, k - 1]
                draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
    # canvas=ImageOps.flip(canvas)# 调整正常的视角
    canvas.save(f"{outdir}/{name}")
    # canvas.getdata()

