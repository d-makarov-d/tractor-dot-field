x, y = brick.get_points(type='REX')
x_all, y_all = brick.get_points()

kernel = gaussian_kde(np.vstack([x, y]))
kernel_all = gaussian_kde(np.vstack([x_all, y_all]))

X, Y = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
positions = np.vstack([X.ravel(), Y.ravel()])

Z = np.reshape(kernel(positions).T, X.shape)
Z_all = np.reshape(kernel_all(positions).T, X.shape)

plt.subplot(221)
plt.title('type REX')
plt.imshow(np.rot90(Z), cmap='seismic', extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar(location='bottom')
plt.subplot(222)
plt.title('type ALL')
plt.imshow(np.rot90(Z_all), cmap='seismic', extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar(location='bottom')
plt.subplot(223)
plt.title('type REX')
plt.scatter(x, y, alpha=0.1, c='r')
plt.gca().set_box_aspect(1)
plt.subplot(224)
plt.title('type ALL')
plt.scatter(x_all, y_all, alpha=0.1, c='r')
plt.gca().set_box_aspect(1)
plt.show()