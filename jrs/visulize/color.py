# rgb
# palette = [[128, 64, 128],
#         [244, 35, 232],
#         [70, 70, 70],
#         [102, 102, 156],
#         [190, 153, 153],
#         [153, 153, 153],
#         [250, 170, 30],
#         [220, 220, 0],
#         [107, 142, 35],
#         [152, 251, 152],
#         [70, 130, 180],
#         [220, 20, 60],
#         [255, 0, 0],
#         [0, 0, 142],
#         [0, 0, 70],
#         [0, 60, 100],
#         [0, 80, 100],
#         [0, 0, 230],
#         [119, 11, 32],
#         [0, 0, 0]] # class 20: ignored, I added it, not cityscapes

palette = [[255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0]] # class 20: ignored, I added it, not cityscapes


my_color = {}
for k, p in enumerate(palette):
        my_color[k+1]=p


palette_bgr = [ [l[2], l[1], l[0]] for l in palette]
             #b            g            r
colorC0 = [(154 / 255, 194 / 255, 182 / 255 )] #blue
colorDe = [(153 / 255, 102 / 255, 0 / 255)] #RGB "#006699"
colorT2 = [(148 / 255, 205 / 255, 127 / 255)] #7FCD94


# colorGD = [(0 / 255, 255 / 255, 255 / 255)] #yellow
colorGD = [(61 / 255, 179 / 255,230 / 255 )] ## E6B33D
colorYellow = [(0 / 255, 255 / 255,255 / 255 )] #

myoc0 = [(154 / 255, 194 / 255,182 / 255 )] #
myode = [(153 / 255, 102 / 255, 0 / 255)] #
myot2 = [(148 / 255, 205 / 255, 127 / 255)] #