import matplotlib.colors as mcolors
import cv2

def call_dictKey(class_num):
    dict_key = []

    for k in mcolors.BASE_COLORS.keys():
        dict_key.append(k)

    for k in mcolors.TABLEAU_COLORS.keys():
        dict_key.append(k)

    for k in mcolors.CSS4_COLORS.keys():
        dict_key.append(k)

    return dict_key[:class_num]

def call_dictValue(class_num):
    dict_value = []

    for v in mcolors.BASE_COLORS.values():
        dict_value.append(v)

    for v in mcolors.TABLEAU_COLORS.values():
        dict_value.append(v)

    for v in mcolors.CSS4_COLORS.values():
        dict_value.append(v)

    return dict_value

def hexa2ten(x):
    return{
        '1' : 1,
        '2' : 2,
        '3' : 3,
        '4' : 4,
        '5' : 5,
        '6' : 6,
        '7' : 7,
        '8' : 8,
        '9' : 9,
        'a' : 10,
        'b' : 11,
        'c' : 12,
        'd' : 13,
        'e' : 14,
        'f' : 15,
    }.get(x, 1)

def hexa2rgb(value, rgb):
    r = value[1:3]
    g = value[3:5]
    b = value[5:7]

    r_value = (hexa2ten(r[0]) * 16) + hexa2ten(r[1])
    g_value = (hexa2ten(g[0]) * 16) + hexa2ten(g[1])
    b_value = (hexa2ten(b[0]) * 16) + hexa2ten(b[1])

    if rgb == True:
        return (r_value, g_value, b_value)
    else:
        return (b_value, g_value, r_value)

def call_rgbValue(class_num, rgb):
    dict_key = call_dictKey(class_num)
    dict_value = call_dictValue(class_num)

    color_map = []

    for i in range(class_num):
        if type(dict_value[i]) == str:
            rgb = hexa2rgb(dict_value[i], rgb)
            color_map.append(rgb)
        elif type(dict_value[i]) == tuple:
            r = dict_value[i][0] * 255
            g = dict_value[i][1] * 255
            b = dict_value[i][2] * 255

            if rgb == True:
                color_map.append((r, g, b))
            else:
                color_map.append((b, g, r))
        else:
            pass

    return color_map

def draw_pixel(predicted_map, class_num = 1):
    height = predicted_map.shape[0]
    width = predicted_map.shape[1]

    copy_map_rgb = cv2.cvtColor(predicted_map.astype('uint8'), cv2.COLOR_GRAY2RGB)

    color_map = call_rgbValue(class_num, True)

    for i in range(height):
        for j in range(width):
            copy_map_rgb[i][j][0] = color_map[predicted_map[i][j]][0]
            copy_map_rgb[i][j][1] = color_map[predicted_map[i][j]][1]
            copy_map_rgb[i][j][2] = color_map[predicted_map[i][j]][2]

    return copy_map_rgb
