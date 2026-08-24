import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook

from facehide.logbook import LogRecord, format_log_line, write_xlsx


class LogbookTests(unittest.TestCase):
    def test_format_includes_time(self) -> None:
        when = datetime(2026, 8, 24, 15, 30, 12)
        line = format_log_line(when, "识别到 Ada（0.86）")
        self.assertTrue(line.startswith("2026-08-24 15:30:12"))
        self.assertIn("Ada", line)

    def test_write_xlsx_time_and_person(self) -> None:
        when = datetime(2026, 8, 24, 15, 30, 12)
        records = [
            LogRecord(when, "摄像头已打开"),
            LogRecord(when, "识别到 Ada（0.86）", person="Ada"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.xlsx"
            write_xlsx(path, records, headers=("时间", "人物", "内容"), sheet="识别日志")
            book = load_workbook(path)
            page = book.active
            self.assertEqual([cell.value for cell in page[1]], ["时间", "人物", "内容"])
            self.assertIn(page["B2"].value, ("", None))
            self.assertEqual(page["B3"].value, "Ada")
            self.assertEqual(page["C3"].value, "识别到 Ada（0.86）")
            self.assertEqual(page["A3"].value, when)


if __name__ == "__main__":
    unittest.main()
