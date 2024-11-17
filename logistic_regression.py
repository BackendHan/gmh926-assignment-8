import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

from sklearn.metrics import log_loss

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)


def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8],
                                  [cluster_std * 0.8, cluster_std]])

    # 生成第一个聚类（类别 0）
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # 生成第二个聚类（类别 1）
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)

    # 将第二个聚类沿 x 和 y 轴位移指定的距离
    X2 += np.array([distance, distance])
    y2 = np.ones(n_samples)

    # 合并两个聚类为一个数据集
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y


# 函数：拟合逻辑回归模型并提取系数
def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2


def do_experiments(start, end, step_num):
    # 设置实验参数
    shift_distances = np.linspace(start, end, step_num)  # 位移距离范围
    beta0_list, beta1_list, beta2_list = [], [], []
    slope_list, intercept_list = [], []
    loss_list, margin_widths = [], []
    sample_data = {}  # 存储用于可视化的样本数据和模型

    n_samples = step_num
    n_cols = 2  # 固定列数
    n_rows = (n_samples + n_cols - 1) // n_cols  # 计算需要的行数
    plt.figure(figsize=(20, n_rows * 10))  # 根据行数调整图像高度

    # 对每个位移距离运行实验
    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)

        # 拟合逻辑回归模型并记录参数
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)

        # 计算决策边界的斜率和截距
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)

        # 计算并存储逻辑损失
        y_pred_proba = model.predict_proba(X)[:, 1]
        loss = log_loss(y, y_pred_proba)
        loss_list.append(loss)

        # 计算 70% 置信度等高线之间的边缘宽度
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # 绘制数据集
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')

        # 绘制决策边界
        x_values = np.array([X[:, 0].min(), X[:, 0].max()])
        y_values = slope * x_values + intercept
        plt.plot(x_values, y_values, label='Decision Boundary', color='green')

        # 绘制淡化的红色和蓝色等高线表示置信水平
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]  # 更高置信水平的透明度更高
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
            if level == 0.7:
                # 计算边缘宽度
                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices,
                                  class_0_contour.collections[0].get_paths()[0].vertices, metric='euclidean')
                min_distance = np.min(distances)
                margin_widths.append(min_distance)

        plt.title(f"Shift Distance = {distance:.2f}", fontsize=24)
        plt.xlabel("x1", fontsize=20)
        plt.ylabel("x2", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # 在图上显示决策边界方程和边缘宽度
        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"
        margin_text = f"Margin Width: {min_distance:.2f}"
        plt.text(x_min + 0.1, y_max - 1.0, equation_text, fontsize=16, color="black", ha='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.text(x_min + 0.1, y_max - 1.5, margin_text, fontsize=16, color="black", ha='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        if i == 1:
            plt.legend(loc='lower right', fontsize=16)

        sample_data[distance] = (X, y, model, beta0, beta1, beta2, min_distance)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")
    plt.close()

    # 绘制参数随位移距离的变化
    plt.figure(figsize=(18, 15))

    # 绘制 beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # 绘制 beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # 绘制 beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # 绘制斜率（beta1 / beta2）
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, marker='o')
    plt.title("Shift Distance vs Beta1 / Beta2 (Slope)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1 / Beta2")
    plt.ylim(-2, 0)

    # 绘制截距比率（-beta0 / beta2）
    plt.subplot(3, 3, 5)
    intercept_ratio = [-b0 / b2 for b0, b2 in zip(beta0_list, beta2_list)]
    plt.plot(shift_distances, intercept_ratio, marker='o')
    plt.title("Shift Distance vs Beta0 / Beta2 (Intercept Ratio)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0 / Beta2")

    # 绘制逻辑损失
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, marker='o')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    # 绘制边缘宽度
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances[:len(margin_widths)], margin_widths, marker='o')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")
    plt.close()


if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
