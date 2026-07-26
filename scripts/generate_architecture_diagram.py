from graphviz import Digraph


def build_diagram() -> Digraph:
    diagram = Digraph(
        name="ml_serving_architecture",
        format="png",
    )

    diagram.attr(
        rankdir="TB",
        splines="ortho",
        nodesep="0.5",
        ranksep="0.7",
        bgcolor="white",
    )

    diagram.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
        fontsize="11",
        margin="0.15",
    )

    diagram.attr(
        "edge",
        fontname="Helvetica",
        fontsize="10",
        arrowsize="0.8",
    )

    diagram.node(
        "dataset",
        "NYC TLC Yellow Taxi Data\nJanuary 2026 Parquet",
        fillcolor="#D9EAF7",
    )

    diagram.node(
        "validation",
        "Schema Validation\nand Data Quality Checks",
        fillcolor="#FFF2CC",
    )

    diagram.node(
        "features",
        "Shared Feature Engineering",
        fillcolor="#FFF2CC",
    )

    diagram.node(
        "training",
        "Model Training\nHistGradientBoostingRegressor",
        fillcolor="#D5E8D4",
    )

    diagram.node(
        "artifacts",
        "Model Artifacts\nfare_model.joblib\nmetrics.json\nmodel_metadata.json",
        fillcolor="#FCE5CD",
    )

    diagram.node(
        "api",
        "Online Serving\nFastAPI /predict",
        fillcolor="#E1D5E7",
    )

    diagram.node(
        "batch",
        "Offline Batch Scoring\nrun_batch_prediction.py",
        fillcolor="#E1D5E7",
    )

    diagram.node(
        "predictor",
        "Shared Serving Layer\nmodel_loader.py + predictor.py",
        fillcolor="#E1D5E7",
    )

    diagram.node(
        "api_output",
        "JSON Prediction Response",
        fillcolor="#E7E6E6",
    )

    diagram.node(
        "batch_output",
        "Scored CSV Output",
        fillcolor="#E7E6E6",
    )

    diagram.edge("dataset", "validation")
    diagram.edge("validation", "features")
    diagram.edge("features", "training")
    diagram.edge("training", "artifacts")

    diagram.edge("artifacts", "predictor")
    diagram.edge("predictor", "api")
    diagram.edge("predictor", "batch")

    diagram.edge("api", "api_output")
    diagram.edge("batch", "batch_output")

    return diagram


def main() -> None:
    diagram = build_diagram()

    output_path = diagram.render(
        filename="docs/architecture",
        cleanup=True,
    )

    print(f"Architecture diagram saved to {output_path}")


if __name__ == "__main__":
    main()