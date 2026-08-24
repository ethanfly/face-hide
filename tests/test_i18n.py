import unittest

from facehide.i18n import _STRINGS, current_language, normalize_language, set_language, t


class I18nTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_language("zh")

    def test_normalize_language(self) -> None:
        self.assertEqual(normalize_language("en-US"), "en")
        self.assertEqual(normalize_language("zh_CN"), "zh")
        self.assertEqual(normalize_language("unknown"), "zh")

    def test_switch_updates_strings(self) -> None:
        set_language("zh")
        self.assertEqual(t("app.name"), "当面隐藏")
        self.assertEqual(t("nav.faces"), "人脸库")
        set_language("en")
        self.assertEqual(current_language(), "en")
        self.assertEqual(t("app.name"), "FaceHide")
        self.assertEqual(t("nav.faces"), "Faces")
        self.assertEqual(t("faces.enable"), "Enable auto-hide")
        self.assertEqual(t("pill.seen", name="Ada"), "Seen Ada")

    def test_format_kwargs(self) -> None:
        set_language("en")
        self.assertEqual(t("pill.faces", count=3), "Faces 3")
        set_language("zh")
        self.assertEqual(t("pill.faces", count=3), "人脸 3")

    def test_zh_en_keys_match(self) -> None:
        self.assertEqual(set(_STRINGS["zh"]), set(_STRINGS["en"]))

    def test_missing_key_falls_back(self) -> None:
        set_language("en")
        self.assertEqual(t("not.a.real.key"), "not.a.real.key")


if __name__ == "__main__":
    unittest.main()
