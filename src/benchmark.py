
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import face_recognition

from database.Sqlite import BASE_DIR
from database.load_faces import load_faces


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


def read_image(path):
    """Read images, including Unicode paths on Windows."""
    data = np.frombuffer(
        Path(path).read_bytes(),
        dtype=np.uint8
    )
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_gallery():
    """Load enrolled face embeddings from SQLite."""
    names, encodings = load_faces()

    if not encodings:
        raise RuntimeError(
            "No enrolled faces. Register people first."
        )

    matrix = np.asarray(
        encodings, dtype=np.float64
    ).reshape(-1, 128)

    return names, matrix


def predict_identity(
    encoding, names, matrix, threshold
):
    """Predict identity from one face embedding."""
    distances = face_recognition.face_distance(
        matrix, encoding
    )

    index = int(np.argmin(distances))
    distance = float(distances[index])

    if distance <= threshold:
        return names[index], distance

    return "Unknown", distance


def collect_samples(dataset_dir):
    """
    Dataset:
        known/PersonName/*.jpg
        unknown/*.jpg

    Unknown may also contain subfolders.
    """
    samples = []

    known_dir = dataset_dir / "known"
    unknown_dir = dataset_dir / "unknown"

    if known_dir.exists():
        for person_dir in sorted(known_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            for path in sorted(person_dir.rglob("*")):
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                ):
                    samples.append(
                        (path, person_dir.name, True)
                    )

    if unknown_dir.exists():
        for path in sorted(unknown_dir.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                samples.append(
                    (path, "Unknown", False)
                )

    return samples


def safe_rate(numerator, denominator):
    if denominator == 0:
        return None

    return round(
        100.0 * numerator / denominator, 4
    )


def evaluate(dataset_dir, threshold=0.5):
    """
    Evaluate open-set face identification.

    Invalid images are excluded from rates and
    counted separately.
    """
    if not 0 < threshold <= 1:
        raise ValueError(
            "Threshold must be in (0, 1]."
        )

    dataset_dir = Path(dataset_dir)
    names, matrix = build_gallery()

    samples = collect_samples(dataset_dir)

    if not samples:
        raise RuntimeError(
            f"No benchmark images found: {dataset_dir}"
        )

    enrolled_names = set(names)

    # Check ground-truth labels before evaluation.
    known_labels = {
        label
        for _, label, is_known in samples
        if is_known
    }

    missing = known_labels - enrolled_names

    if missing:
        raise ValueError(
            "Known identities not found in database: "
            + ", ".join(sorted(missing))
        )

    results = []

    correct = 0
    valid = 0
    invalid = 0

    known_total = 0
    unknown_total = 0

    false_reject = 0
    false_accept = 0
    wrong_identity = 0

    latencies = []

    for path, expected, is_known in samples:
        start = time.perf_counter()

        predicted = ""
        distance = None
        status = "invalid"

        try:
            rgb = read_image(path)

            locations = face_recognition.face_locations(
                rgb, model="hog"
            )

            if len(locations) != 1:
                raise ValueError(
                    f"Expected 1 face, found {len(locations)}"
                )

            encodings = face_recognition.face_encodings(
                rgb, locations
            )

            if len(encodings) != 1:
                raise ValueError(
                    "Face encoding failed"
                )

            predicted, distance = predict_identity(
                encodings[0],
                names,
                matrix,
                threshold
            )

            status = "correct" if (
                predicted == expected
            ) else "incorrect"

            valid += 1

            if status == "correct":
                correct += 1

            if is_known:
                known_total += 1

                if predicted == "Unknown":
                    false_reject += 1
                elif predicted != expected:
                    wrong_identity += 1

            else:
                unknown_total += 1

                if predicted != "Unknown":
                    false_accept += 1

        except (ValueError, OSError) as exc:
            invalid += 1
            status = f"invalid: {exc}"

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        if not status.startswith("invalid"):
            latencies.append(elapsed_ms)

        results.append({
            "file": str(path),
            "expected": expected,
            "predicted": predicted,
            "distance": (
                round(distance, 6)
                if distance is not None
                else ""
            ),
            "latency_ms": round(elapsed_ms, 3),
            "status": status
        })

        print(
            f"[{len(results)}/{len(samples)}] "
            f"{path.name}: {status}"
        )

    mean_latency = (
        float(np.mean(latencies))
        if latencies
        else None
    )

    p95_latency = (
        float(np.percentile(latencies, 95))
        if latencies
        else None
    )

    summary = {
        "threshold": threshold,
        "gallery_embeddings": len(names),
        "gallery_identities": len(enrolled_names),
        "total_images": len(samples),
        "valid_images": valid,
        "invalid_images": invalid,
        "known_test_images": known_total,
        "unknown_test_images": unknown_total,
        "correct_predictions": correct,
        "false_rejections": false_reject,
        "false_acceptances": false_accept,
        "wrong_identity_predictions": wrong_identity,
        "accuracy_percent": safe_rate(correct, valid),
        "far_percent": safe_rate(
            false_accept, unknown_total
        ),
        "frr_percent": safe_rate(
            false_reject, known_total
        ),
        "wrong_identity_percent": safe_rate(
            wrong_identity, known_total
        ),
        "mean_latency_ms": (
            round(mean_latency, 3)
            if mean_latency is not None
            else None
        ),
        "p95_latency_ms": (
            round(p95_latency, 3)
            if p95_latency is not None
            else None
        )
    }

    return summary, results


def save_reports(summary, results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    detail_path = output_dir / (
        f"benchmark_details_{timestamp}.csv"
    )
    summary_path = output_dir / (
        f"benchmark_summary_{timestamp}.json"
    )
    summary_csv_path = output_dir / (
        f"benchmark_summary_{timestamp}.csv"
    )

    with detail_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file", "expected", "predicted",
                "distance", "latency_ms", "status"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    with summary_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file,
            indent=2,
            ensure_ascii=False
        )

    with summary_csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys())
        )
        writer.writeheader()
        writer.writerow(summary)

    return (
        detail_path,
        summary_path,
        summary_csv_path
    )


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Benchmark"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=BASE_DIR / "data" / "benchmark"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "reports"
    )

    args = parser.parse_args()

    summary, results = evaluate(
        args.dataset,
        args.threshold
    )

    paths = save_reports(
        summary,
        results,
        args.output
    )

    print("\n=== BENCHMARK SUMMARY ===")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nReports saved:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
