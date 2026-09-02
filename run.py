from naxi.v_0d2.gridman.train import train_model, dist


def main(is_sft: bool = False):
    try:
        train_model(is_sft)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main(is_sft=False)
