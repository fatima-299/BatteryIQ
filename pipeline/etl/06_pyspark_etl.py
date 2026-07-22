"""
BatteryIQ — Step 11: PySpark ETL Pipeline (Windows-compatible)
===============================================================
Reads feature matrix with Spark, runs distributed analytics,
writes outputs as CSV (Windows-compatible, no Hadoop native IO needed).

Run from BatteryIQ root:
  python pipeline/etl/06_pyspark_etl.py
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

ROOT     = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "features"
OUT_DIR  = FEAT_DIR / "spark_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Spark Session ──────────────────────────────────────────────────────────
def create_spark() -> SparkSession:
    print("🔥 Initialising Spark session ...")
    spark = (
        SparkSession.builder
        .appName("BatteryIQ-ETL")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        # Windows fix — disable native Hadoop IO
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"   ✅ Spark {spark.version} | Cores: {spark.sparkContext.defaultParallelism}")
    return spark


# ── Extract ────────────────────────────────────────────────────────────────
def read_data(spark: SparkSession):
    print(f"\n📂 Reading feature matrix ...")
    df = spark.read.csv(
        str(FEAT_DIR / "feature_matrix.csv"),
        header=True, inferSchema=True, nullValue="nan"
    )
    print(f"   ✅ {df.count():,} rows × {len(df.columns)} cols")
    return df


# ── Transform ──────────────────────────────────────────────────────────────
def transform(df):
    print("\n⚙️  Applying Spark transformations ...")

    # Degradation category
    df = df.withColumn(
        "degradation_category",
        F.when(F.col("soh_pct") >= 95, "excellent")
         .when(F.col("soh_pct") >= 90, "good")
         .when(F.col("soh_pct") >= 80, "fair")
         .when(F.col("soh_pct") >= 70, "poor")
         .otherwise("critical")
    )

    # Risk score
    df = df.withColumn(
        "risk_score",
        F.round(
            F.when(F.col("soh_pct") >= 95, (100 - F.col("soh_pct")) * 1.0)
             .when(F.col("soh_pct") >= 80, (100 - F.col("soh_pct")) * 2.0)
             .otherwise((100 - F.col("soh_pct")) * 4.0), 2
        )
    )

    # Alert flag
    df = df.withColumn(
        "alert_flag",
        F.when(F.col("soh_pct") < 80, "EOL_REACHED")
         .when(F.col("soh_pct") < 85, "WARNING")
         .when(F.col("soh_pct") < 90, "MONITOR")
         .otherwise("OK")
    )

    # Window functions
    window_cell = Window.partitionBy("cell_id").orderBy("cycle_number")
    df = df.withColumn(
        "cumulative_min_soh",
        F.min("soh_pct").over(
            window_cell.rowsBetween(Window.unboundedPreceding, 0)
        )
    )
    df = df.withColumn("cycle_rank", F.row_number().over(window_cell))

    print(f"   ✅ Added: degradation_category, risk_score, alert_flag,")
    print(f"      cumulative_min_soh, cycle_rank (Spark window functions)")
    return df


# ── Spark SQL Analytics ────────────────────────────────────────────────────
def run_spark_sql(spark, df):
    print("\n📊 Running Spark SQL analytics ...")
    df.createOrReplaceTempView("battery_cycles")

    queries = {
        "Fleet Health by Source": """
            SELECT source, chemistry,
                COUNT(DISTINCT cell_id)   AS n_cells,
                COUNT(*)                  AS n_cycles,
                ROUND(AVG(soh_pct),2)    AS avg_soh,
                ROUND(MIN(soh_pct),2)    AS min_soh,
                ROUND(MAX(soh_pct),2)    AS max_soh,
                SUM(CASE WHEN alert_flag='EOL_REACHED' THEN 1 ELSE 0 END) AS eol_cycles,
                SUM(CASE WHEN alert_flag='WARNING' THEN 1 ELSE 0 END)     AS warning_cycles
            FROM battery_cycles
            GROUP BY source, chemistry
            ORDER BY source
        """,
        "Top 10 Most Degraded Cells": """
            SELECT cell_id, source, chemistry,
                ROUND(MIN(soh_pct),2)    AS min_soh,
                MAX(cycle_number)        AS total_cycles,
                ROUND(AVG(risk_score),1) AS avg_risk
            FROM battery_cycles
            GROUP BY cell_id, source, chemistry
            ORDER BY min_soh ASC LIMIT 10
        """,
        "Degradation Category Distribution": """
            SELECT degradation_category,
                COUNT(*) AS n_cycles,
                ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),2) AS pct
            FROM battery_cycles
            GROUP BY degradation_category
            ORDER BY CASE degradation_category
                WHEN 'excellent' THEN 1 WHEN 'good' THEN 2
                WHEN 'fair' THEN 3 WHEN 'poor' THEN 4 ELSE 5 END
        """,
        "SOH by Lifecycle Stage": """
            SELECT
                CASE lifecycle_stage
                    WHEN 0 THEN 'Early (0-33%)'
                    WHEN 1 THEN 'Mid (33-66%)'
                    WHEN 2 THEN 'Late (66-100%)'
                END AS stage,
                COUNT(*) AS n_cycles,
                ROUND(AVG(soh_pct),2) AS avg_soh,
                ROUND(AVG(risk_score),2) AS avg_risk
            FROM battery_cycles
            WHERE lifecycle_stage IS NOT NULL
            GROUP BY lifecycle_stage ORDER BY lifecycle_stage
        """
    }

    results = {}
    for name, query in queries.items():
        print(f"\n   {name}:")
        result = spark.sql(query)
        result.show(truncate=False)
        results[name] = result

    return results


# ── Load — write using pandas (Windows-compatible) ─────────────────────────
def write_outputs(df, spark, results):
    print("\n💾 Writing outputs (CSV via pandas — Windows compatible) ...")

    # Convert Spark DataFrame to pandas and save
    print("   Converting to pandas ...")
    pandas_df = df.toPandas()

    # 1. Full enriched feature matrix
    out1 = OUT_DIR / "feature_matrix_enriched.csv"
    pandas_df.to_csv(out1, index=False)
    print(f"   ✅ Enriched feature matrix → {out1.name} ({len(pandas_df):,} rows)")

    # 2. Fleet summary per cell (for Power BI)
    fleet_summary = (
        pandas_df.groupby(["cell_id", "source", "chemistry"])
        .agg(
            n_cycles         = ("soh_pct", "count"),
            avg_soh          = ("soh_pct", "mean"),
            min_soh          = ("soh_pct", "min"),
            max_soh          = ("soh_pct", "max"),
            final_soh        = ("soh_pct", "last"),
            total_cycles     = ("cycle_number", "max"),
            avg_risk_score   = ("risk_score", "mean"),
            final_alert      = ("alert_flag", "last"),
            final_category   = ("degradation_category", "last"),
        )
        .round(3)
        .reset_index()
    )
    out2 = OUT_DIR / "fleet_summary_per_cell.csv"
    fleet_summary.to_csv(out2, index=False)
    print(f"   ✅ Fleet summary per cell → {out2.name} ({len(fleet_summary)} cells)")

    # 3. Analytics summary tables
    import pandas as pd
    all_analytics = []
    for name, spark_df in results.items():
        pdf = spark_df.toPandas()
        pdf.insert(0, "query", name)
        all_analytics.append(pdf)

    out3 = OUT_DIR / "spark_sql_analytics.csv"
    pd.concat(all_analytics).to_csv(out3, index=False)
    print(f"   ✅ Spark SQL analytics → {out3.name}")

    # 4. PostgreSQL load attempt
    load_to_postgres(pandas_df)

    return fleet_summary


# ── PostgreSQL load ────────────────────────────────────────────────────────
def load_to_postgres(pandas_df):
    print("\n🐘 Attempting PostgreSQL load ...")
    try:
        import psycopg2

        conn = psycopg2.connect(
            host="localhost", port=5432,
            database="batteryiq",
            user="postgres", password="postgres"
        )
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS battery_cycles (
                cell_id              VARCHAR(50),
                source               VARCHAR(20),
                chemistry            VARCHAR(10),
                cycle_number         INTEGER,
                soh_pct              FLOAT,
                rul_cycles           INTEGER,
                cycle_capacity_ah    FLOAT,
                internal_resistance  FLOAT,
                avg_temp_c           FLOAT,
                soh_lag_1            FLOAT,
                soh_roll_mean_10     FLOAT,
                capacity_fade_rate   FLOAT,
                arrhenius_factor     FLOAT,
                cycle_normalized     FLOAT,
                lifecycle_stage      INTEGER,
                degradation_category VARCHAR(20),
                risk_score           FLOAT,
                alert_flag           VARCHAR(20),
                cumulative_min_soh   FLOAT
            );
        """)
        cur.execute("TRUNCATE TABLE battery_cycles;")
        conn.commit()

        cols = ["cell_id","source","chemistry","cycle_number","soh_pct",
                "rul_cycles","cycle_capacity_ah","internal_resistance",
                "avg_temp_c","soh_lag_1","soh_roll_mean_10",
                "capacity_fade_rate","arrhenius_factor","cycle_normalized",
                "lifecycle_stage","degradation_category","risk_score",
                "alert_flag","cumulative_min_soh"]
        cols = [c for c in cols if c in pandas_df.columns]
        data = pandas_df[cols]

        chunk_size = 5000
        for i in range(0, len(data), chunk_size):
            chunk  = data.iloc[i:i+chunk_size]
            values = [tuple(r) for r in chunk.itertuples(index=False)]
            ph     = ','.join(['%s'] * len(cols))
            cur.executemany(
                f"INSERT INTO battery_cycles ({','.join(cols)}) VALUES ({ph})",
                values
            )
            conn.commit()
            if (i // chunk_size) % 10 == 0:
                print(f"   Inserted {min(i+chunk_size,len(data)):,}/{len(data):,} rows ...")

        cur.close()
        conn.close()
        print(f"   ✅ Loaded {len(data):,} rows → PostgreSQL batteryiq.battery_cycles")

    except ImportError:
        print("   ⚠️  psycopg2 not installed → skipping")
        print("   Run: pip install psycopg2-binary")
    except Exception as e:
        print(f"   ⚠️  PostgreSQL not running: {type(e).__name__}")
        print(f"   → Install PostgreSQL for Power BI connection (Step 16)")
        print(f"   → CSV outputs are ready and equivalent for ML models")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n🔋 BatteryIQ — PySpark ETL Pipeline")
    print("=" * 55)

    spark   = create_spark()
    df      = read_data(spark)       # Extract
    df      = transform(df)          # Transform
    results = run_spark_sql(spark, df)
    fleet   = write_outputs(df, spark, results)  # Load

    print("\n" + "=" * 55)
    print("✅ PySpark ETL complete!")
    print(f"   Outputs → data/features/spark_output/")
    print(f"     • feature_matrix_enriched.csv  ({len(fleet)} cells processed)")
    print(f"     • fleet_summary_per_cell.csv   (Power BI ready)")
    print(f"     • spark_sql_analytics.csv      (Chapter 4 tables)")
    print(f"\n   Next: python ml/training/07_xgboost_model.py")

    spark.stop()


if __name__ == "__main__":
    main()
