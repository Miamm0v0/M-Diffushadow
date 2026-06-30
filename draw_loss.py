import json
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
import argparse
import numpy as np

def main(json_path, output_plot_path=None, title="Training & Validation Loss Curves", plot_last_n_epochs=None, min_epochs=10, num_qubits=None, selected_runs=None, ground_truth=False):
    if output_plot_path is None:
        output_plot_path = json_path.replace('.json', '_plot.png')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON log file not found: {json_path}")

    with open(json_path, 'r') as f:
        logs = json.load(f)

    print(f"✅ Loaded {len(logs)} training runs from {json_path}")

    valid_logs = []
    for i, log in enumerate(logs):
        epochs = log.get("total_epochs", 0)
        if epochs < min_epochs:
            print(f"⚠️  Run {i+1}: Skipped (only {epochs} epochs < {min_epochs})")
            continue
        if not log.get("epoch_losses"):
            print(f"⚠️  Run {i+1}: Skipped (no loss data)")
            continue
        valid_logs.append(log)

    print(f"📊 Plotting {len(valid_logs)} valid training runs...")

    plt.figure(figsize=(14, 9))

    # 使用 colormap 生成多样颜色
    cmap = plt.get_cmap('tab20')  # 推荐：'tab10', 'Set1', 'viridis', 'plasma'
    colors = [cmap(i % 20) for i in range(len(valid_logs))]  # tab10 有 10 种高对比度颜色

    if selected_runs is None:
        # 如果没有指定，则默认处理所有有效日志
        runs_to_plot = valid_logs
    else:
        # 仅选取用户指定的运行
        runs_to_plot = [log for idx, log in enumerate(valid_logs) if (idx+1) in selected_runs]
        print(f"Selected {len(runs_to_plot)} runs to plot based on your selection.")    

    for idx, log in enumerate(runs_to_plot):
        train_losses = log["epoch_losses"]
        eval_losses = log.get("test_losses", [])  # 建议统一字段名：eval_losses 或 test_losses

        if plot_last_n_epochs:
            train_losses = train_losses[-plot_last_n_epochs:]
            eval_losses = eval_losses[-plot_last_n_epochs:] if eval_losses else []
            epochs = range(len(train_losses))
            xlabel = f"Epoch (Last {plot_last_n_epochs})"
        else:
            epochs = range(1, len(train_losses) + 1)
            xlabel = "Epoch"

        args = log.get("args", {})
        lr = args.get("lr", "unknown_lr")
        scheduler = args.get("lr_scheduler", "unknown")
        eps = args.get("eps", 0.001)
        # run_name = f"Run {idx+1}: LR={lr}, {scheduler}, eps={eps}"
        run_name = f"Diffushadow"

        color = colors[idx]  # 为每个 run 分配一个颜色

        # 训练损失：实线
        plt.plot(epochs, train_losses, label=f"{run_name} (Train)", 
                 color=color, linewidth=2.5, alpha=0.9)

        # 验证损失：同色，虚线
        if eval_losses and len(eval_losses) > 0:
            min_len = min(len(train_losses), len(eval_losses))
            plt.plot(epochs[-min_len:], eval_losses[-min_len:], 
                     label=f"{run_name} (Eval)", 
                     color=color, linewidth=2.5, linestyle='--', alpha=0.9)
        else:
            print(f"🔍 Run {idx+1}: No eval loss data found.")

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel("Loss", fontsize=24)
    plt.title(title, fontsize=32)
    plt.grid(True, alpha=0.3)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    # -------------------------------
    # ✅ 添加 Ground Truth 曲线（始终显示）
    # -------------------------------
    if ground_truth:
        ground_truth_epochs = list(range(1, 64))  # 1 到 63 轮

        if num_qubits == 10:
            ground_train_losses = [
                5.890209, 5.449035, 5.402958, 5.402752, 5.387996,
                5.372346, 5.363989, 5.376809, 5.371220, 5.364058,
                5.359056, 5.352513, 5.348057, 5.343423, 5.340616,
                5.358129, 5.356683, 5.353372, 5.350655, 5.347749,
                5.345740, 5.342632, 5.340187, 5.337427, 5.335215,
                5.332857, 5.329865, 5.327913, 5.325843, 5.324858,
                5.323817, 5.342203, 5.342000, 5.340319, 5.339500,
                5.337862, 5.335409, 5.333685, 5.332985, 5.330718,
                5.329141, 5.326864, 5.325040, 5.323385, 5.321500,
                5.319071, 5.316975, 5.314237, 5.313201, 5.310622,
                5.309242, 5.307874, 5.304963, 5.303324, 5.302155,
                5.301491, 5.298901, 5.298381, 5.296792, 5.297219,
                5.294323, 5.293900, 5.293895
            ]
            ground_train_losses = [loss / 10 for loss in ground_train_losses]

            ground_test_losses = [
                5.556294, 5.429984, 5.393864, 5.405584, 5.383200,
                5.377009, 5.368868, 5.370459, 5.367722, 5.372553,
                5.366290, 5.361434, 5.360430, 5.359025, 5.357013,
                5.374488, 5.366825, 5.362669, 5.364116, 5.372996,
                5.364043, 5.360903, 5.363252, 5.368446, 5.363262,
                5.361973, 5.360668, 5.362340, 5.359559, 5.360562,
                5.361134, 5.367551, 5.365781, 5.371909, 5.365823,
                5.364760, 5.377406, 5.375705, 5.381417, 5.370844,
                5.367194, 5.370270, 5.376934, 5.368407, 5.372310,
                5.374086, 5.371150, 5.377394, 5.375198, 5.383675,
                5.373901, 5.377960, 5.381534, 5.381374, 5.379888,
                5.379410, 5.381927, 5.381400, 5.382698, 5.382616,
                5.385189, 5.384831, 5.384528
            ]
            ground_test_losses = [loss / 10 for loss in ground_test_losses]
        elif num_qubits == 14:
            ground_train_losses = [
                8.218975, 7.741343, 7.693802, 7.695484, 7.684557,
                7.673220, 7.665827, 7.679430, 7.674179, 7.672261,
                7.667558, 7.664193, 7.661215, 7.658972, 7.657085,
                7.669052, 7.668408, 7.666442, 7.664792, 7.664076,
                7.662190, 7.660404, 7.658700, 7.657136, 7.655534,
                7.653062, 7.652021, 7.651252, 7.649673, 7.648714,
                7.648343, 7.660079, 7.660336, 7.659773, 7.658336,
                7.657285, 7.657485, 7.655653, 7.654755, 7.653696,
                7.652305, 7.650910, 7.649043, 7.648366, 7.645886,
                7.643724, 7.642728, 7.640242, 7.639539, 7.638054,
                7.635910, 7.634042, 7.632028, 7.632213, 7.629489,
                7.628696, 7.627554, 7.625531, 7.625232, 7.624724,
                7.623241, 7.623255, 7.622597
            ]
            ground_train_losses = [loss / 14 for loss in ground_train_losses]
        # print(ground_train_losses)
            ground_test_losses = [
                7.760520, 7.684929, 7.673476, 7.674230, 7.676937,
                7.667745, 7.663925, 7.676011, 7.669036, 7.675438,
                7.667503, 7.669369, 7.666380, 7.663435, 7.663923,
                7.670518, 7.671121, 7.667796, 7.667116, 7.668753,
                7.668723, 7.667721, 7.669245, 7.669576, 7.668255,
                7.667507, 7.668920, 7.668474, 7.669126, 7.669416,
                7.669382, 7.668545, 7.666618, 7.671946, 7.682299,
                7.674411, 7.672181, 7.677629, 7.673567, 7.674561,
                7.675544, 7.675840, 7.676204, 7.677555, 7.680774,
                7.681325, 7.686830, 7.684104, 7.683704, 7.682647,
                7.687673, 7.688151, 7.689326, 7.690515, 7.688763,
                7.690127, 7.692101, 7.693741, 7.691437, 7.693529,
                7.693489, 7.693855, 7.694021
            ]    
            ground_test_losses = [loss / 14 for loss in ground_test_losses]


        # print(ground_test_losses)

        # 根据当前绘图范围裁剪 ground truth
        if plot_last_n_epochs:
            start_idx = max(0, len(ground_truth_epochs) - plot_last_n_epochs)
            epochs_plot = ground_truth_epochs[start_idx:]
            train_gt_plot = ground_train_losses[start_idx:]
            test_gt_plot = ground_test_losses[start_idx:]
        else:
            epochs_plot = ground_truth_epochs
            train_gt_plot = ground_train_losses
            test_gt_plot = ground_test_losses

        # 绘制 Ground Truth
        plt.plot(epochs_plot, train_gt_plot,
                label='ShadowGPT (Train)', 
                color='red', linewidth=3, alpha=0.9, linestyle='-', marker='.', markersize=4)

        plt.plot(epochs_plot, test_gt_plot,
                label='ShadowGPT (Eval)', 
                color='red', linewidth=3, alpha=0.9, linestyle='--', marker='s', markersize=4)



    plt.legend(bbox_to_anchor=(0.72, 0.98), loc='upper left', fontsize=14)
    
    if plot_last_n_epochs is None:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Plot saved to: {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training and validation loss curves.')
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--selected_runs', type=int, default=None, nargs='+',
                        help='Specify which runs to plot (1-based indices). E.g., --selected_runs 1 3 5')
    parser.add_argument('--output', type=str)
    parser.add_argument('--title', type=str, default="Training & Validation Loss Curves")
    parser.add_argument('--last_n', type=int)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--num_qubits', type=int, default=10, help='Number of qubits in the model')
    parser.add_argument('--ground_truth', action="store_true", help='whether to plot the ground truth curve')
    args = parser.parse_args()
    main(args.json_path, args.output, args.title, args.last_n, args.min_epochs, args.num_qubits, args.selected_runs, args.ground_truth)