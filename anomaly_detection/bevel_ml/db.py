"""
bevel_ml.db
~~~~~~~~~~~

"""
import os
import sqlite3
from typing import Union, Literal
import pandas as pd
import pandas.io.sql as psql
from pathlib import Path
from pandas import DataFrame


def _get_connection(db_path: Union[Path, str]) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    return conn


def _executescript_on_conn(conn: sqlite3.Connection, sql: str) -> None:
    cur = conn.cursor()
    cur.executescript(sql)
    conn.commit()
    cur.close()


def _executescript(db_path: Union[Path, str], sql: str) -> None:
    with _get_connection(db_path) as conn:
        _executescript_on_conn(conn, sql)


DROP_TABLE_SQL_BASE = """
DROP TABLE IF EXISTS "{tbl_name}";
"""


def drop_table(db_path: Union[Path, str], tbl_name: str) -> None:
    sql = DROP_TABLE_SQL_BASE.format(tbl_name=tbl_name)
    _executescript(db_path, sql)


def load_df(
        db_path: Union[str, Path],
        tbl_name: str,
        columns: list[str] = None,
        dtype: dict[str, any] = None,
        parse_dates=None,
):
    """DataFrameをDBから読み込む。
    Args:
        db_path: データベースファイルのパス
        tbl_name: テーブル名
        columns: 結果として返すDataFrameの列名
        dtype: 結果として返すDataFrameの型。{列名: 型}のdict形式。
        parse_dates:
    Returns:
        DataFrame:
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f'db_path is not exist.  \n    db_path={str(db_path)}')

    columns = '*' if columns is None else ','.join(columns)
    sql = f'select {columns} from "{tbl_name}"'
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(sql, con=con, dtype=dtype, parse_dates=parse_dates)
    return df


def save_df(
        df: DataFrame,
        db_path: Union[str, Path],
        tbl_name: str,
        if_exists: Literal['fail', 'replace', 'append'] = 'replace',
) -> None:
    """DataFrameをDBのテーブルとして保存する。
    Args:
        df: 保存対象のDataFrame
        db_path: データベースファイルのパス
        tbl_name: テーブル名
        if_exists:
    Returns:
        None
    """
    os.makedirs(db_path.parent, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        psql.to_sql(df, tbl_name, conn, if_exists=if_exists, index=False)
