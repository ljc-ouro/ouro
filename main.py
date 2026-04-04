from naxi.v_0d1.gridman.train import train_model, generate_test, dist
from naxi.v_0d1.gridman.chat import gridman_chat


def main(mode: str = 'chat', is_sft: bool = False, test_only: bool = False):
    if mode == 'train':
        if test_only:
            generate_test(is_sft=is_sft)
        else:
            try:
                train_model(is_sft=is_sft)
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()

    if mode == 'chat':
        gridman_chat()


if __name__ == '__main__':
    main()