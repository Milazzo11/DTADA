"""
Test package.

:author: Max Milazzo
:email: mam9563@rit.edu
"""


from DTAda.train import train
from DTAda.predict import predict


def main() -> None:
    """
    Tests.
    """
    
    print("DT TRAIN:\n")
    dt_model, dt_accuracy = train("data.txt", "dt", 5, 10, "dt.pkl", 0.1)
    # train(data_file: str, mode: str, depth: int = 5, num_models: int = 10,
    #    save_file: str = None, test_set_faction: float = 0.1)
    
    print("\nADA TRAIN:\n")
    ada_model, ada_accuracy = train("data.txt", "ada", 5, 10, "ada.pkl", 0.1)
    # train(data_file: str, mode: str, depth: int = 5, num_models: int = 10,
    #    save_file: str = None, test_set_faction: float = 0.1)
    
    print("\nDT RUN:")
    print("PREDICTION:", dt_model.run(
        [False, False, True, False, False, False, False, True]
    ))
    # model.run(prediction_atts: list)
    
    print("PREDICTION:", predict(
        "dt.pkl", [False, False, True, False, False, False, False, True]
    ))
    # predict(model_file: str, prediction_atts: list)
    
    print("\nADA RUN:")
    print("PREDICTION:", ada_model.run(
        [False, False, True, False, False, False, False, True]
    ))
    # model.run(prediction_atts: list)
    
    print("PREDICTION:", predict(
        "ada.pkl", [False, False, True, False, False, False, False, True]
    ))
    # predict(model_file: str, prediction_atts: list)


if __name__ == "__main__":
    main()