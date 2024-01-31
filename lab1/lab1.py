import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from skimage.transform import radon, iradon


print("1...")
img = io.imread("input_image.png")
gray_img = color.rgb2gray(img)

print("2...")
plt.imshow(img)
plt.title("Оригинал")
plt.show()

print("3...")
plt.imshow(gray_img, cmap="gray")
plt.title("Ч/Б")
plt.show()

print("4...")

for step_deg in [10., 5., 1.]:
    print(f"    {step_deg} deg...")

    n_points = int(round(180. / step_deg))
    theta = np.linspace(0., 180., n_points, endpoint=False)
    radon_data = radon(gray_img, theta=theta)
    restored_image = iradon(radon_data, theta)

    plt.imshow(restored_image, cmap="gray")
    plt.title(f"Восстановленное изображение (шаг: {step_deg}°)")
    plt.show()

print("Done!")
