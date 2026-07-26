from src.serving.model_loader import load_model


def main():
    model = load_model()
    print(type(model))


if __name__ == "__main__":
    main()