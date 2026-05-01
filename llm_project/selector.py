from critic import score_reasonings


def select_best(paths):
    evaluations = score_reasonings(paths)
    evaluation_by_path = {item["path"]: item for item in evaluations}
    results = []

    for index, reasoning in enumerate(paths, start=1):
        evaluation = evaluation_by_path.get(
            index,
            {"score": 5.0, "summary": "Evaluation unavailable"},
        )
        results.append(
            {
                "id": index,
                "reasoning": reasoning,
                "score": evaluation["score"],
                "summary": evaluation["summary"],
            }
        )

    best = max(results, key=lambda item: item["score"]) if results else None

    return results, best