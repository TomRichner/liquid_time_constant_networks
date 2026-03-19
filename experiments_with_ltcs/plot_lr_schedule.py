"""Visualize the warmup-hold-cosine learning rate schedule.

Usage: python3 plot_lr_schedule.py [--epochs 20] [--batches_per_epoch 460]
"""
import sys
sys.path.insert(0, '.')
from lr_schedule import warmup_hold_cosine_lr
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batches_per_epoch', type=int, default=460,
                        help='HAR: 7352 train / 16 batch = 459')
    parser.add_argument('--output', default='lr_schedule.png')
    args = parser.parse_args()

    total_steps = args.epochs * args.batches_per_epoch
    steps = list(range(total_steps))
    lrs = [warmup_hold_cosine_lr(s, total_steps) for s in steps]

    # Print key points
    warmup_end = int(0.20 * total_steps)
    hold_end = int(0.70 * total_steps)
    print(f"Total steps: {total_steps}")
    print(f"Warmup: 0 → {warmup_end} (epoch {warmup_end/args.batches_per_epoch:.1f})")
    print(f"Hold:   {warmup_end} → {hold_end} (epoch {hold_end/args.batches_per_epoch:.1f})")
    print(f"Decay:  {hold_end} → {total_steps} (epoch {hold_end/args.batches_per_epoch:.1f} → {args.epochs})")
    print(f"LR range: {lrs[0]:.2e} → {max(lrs):.2e} → {lrs[-1]:.2e}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        epochs_axis = [s / args.batches_per_epoch for s in steps]
        ax.plot(epochs_axis, lrs, 'b-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(f'Warmup-Hold-Cosine LR Schedule ({args.epochs} epochs, {args.batches_per_epoch} batches/epoch)', fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Mark phase boundaries
        warmup_epoch = warmup_end / args.batches_per_epoch
        hold_epoch = hold_end / args.batches_per_epoch
        ax.axvline(warmup_epoch, color='green', linestyle='--', alpha=0.7, label=f'Warmup end ({warmup_epoch:.1f})')
        ax.axvline(hold_epoch, color='red', linestyle='--', alpha=0.7, label=f'Decay start ({hold_epoch:.1f})')
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"\nSaved: {args.output}")
    except ImportError:
        print("\nmatplotlib not available — printed values only")

if __name__ == '__main__':
    main()
