from naxi.v_0d2.gridman.chat import gridman_chat


def main():
    print('Choose model: 1. Pre-train 2. Sft')
    is_sft = input('Input the number and enter: ').startswith('2')

    gridman_chat(is_sft)

    print('Thanks!')


if __name__ == '__main__':
    main()
