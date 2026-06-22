#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import importlib.util
import os
import sys
from unittest import mock


_MOCK_MODULES = [
    "numpy",
    "np",
    "pdfplumber",
    "xgboost",
    "xgb",
    "huggingface_hub",
    "PIL",
    "PIL.Image",
    "pypdf",
    "sklearn",
    "sklearn.cluster",
    "sklearn.metrics",
    "common",
    "common.constants",
    "common.file_utils",
    "common.misc_utils",
    "common.settings",
    "common.token_utils",
    "deepdoc",
    "deepdoc.vision",
    "deepdoc.parser",
    "deepdoc.parser.utils",
    "rag",
    "rag.nlp",
    "rag.prompts",
    "rag.prompts.generator",
]
for _m in _MOCK_MODULES:
    if _m not in sys.modules:
        sys.modules[_m] = mock.MagicMock()


def _find_project_root(marker="pyproject.toml"):
    cur = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(cur, marker)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError(f"Could not locate project root (missing {marker})")
        cur = parent


_MODULE_PATH = os.path.join(_find_project_root(), "deepdoc", "parser", "pdf_parser.py")
_SPEC = importlib.util.spec_from_file_location("pdf_parser", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

_Parser = _MOD.RAGFlowPdfParser


def _build_parser(boxes):
    parser = object.__new__(_Parser)
    parser.boxes = boxes
    parser.mean_height = [10]
    parser.mean_width = [10]
    parser.is_english = False
    return parser


def test_naive_vertical_merge_keeps_table_rows_separate():
    parser = _build_parser(
        [
            {
                "text": "row1",
                "x0": 10,
                "x1": 60,
                "top": 10,
                "bottom": 20,
                "page_number": 1,
                "layout_type": "table",
                "layoutno": "table-0",
                "R": 0,
                "R_top": 10,
                "R_bott": 20,
            },
            {
                "text": "row2",
                "x0": 10,
                "x1": 60,
                "top": 22,
                "bottom": 32,
                "page_number": 1,
                "layout_type": "table",
                "layoutno": "table-0",
                "R": 1,
                "R_top": 22,
                "R_bott": 32,
            },
        ]
    )

    parser._naive_vertical_merge()

    assert len(parser.boxes) == 2
    assert [b["text"] for b in parser.boxes] == ["row1", "row2"]


def test_naive_vertical_merge_still_merges_plain_text():
    parser = _build_parser(
        [
            {
                "text": "hello",
                "x0": 10,
                "x1": 60,
                "top": 10,
                "bottom": 20,
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "text-0",
            },
            {
                "text": "world",
                "x0": 12,
                "x1": 58,
                "top": 22,
                "bottom": 32,
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "text-0",
            },
        ]
    )

    parser._naive_vertical_merge()

    assert len(parser.boxes) == 1
    assert parser.boxes[0]["text"] == "hello world"
