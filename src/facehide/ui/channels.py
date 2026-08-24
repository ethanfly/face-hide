from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from facehide.config import AUTH_MODES, CHANNEL_KINDS, KvPair, MessageChannel
from facehide.gallery import new_id
from facehide.i18n import t
from facehide.ui.icons import app_icon
from facehide.ui.styles import APP_QSS, apply_dark_surface


def kind_label(kind: str) -> str:
    return t(f"channel.kind.{kind}") if kind in CHANNEL_KINDS else kind


def auth_label(mode: str) -> str:
    return t(f"channel.auth.{mode}") if mode in AUTH_MODES else mode


def channel_summary(channel: MessageChannel) -> str:
    state = t("channel.on") if channel.enabled else t("channel.off")
    return f"{channel.name}\n{kind_label(channel.kind)}  ·  {state}"


def parse_pairs(text: str) -> list[KvPair]:
    pairs: list[KvPair] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        if key:
            pairs.append(KvPair(key=key, value=value.strip()))
    return pairs


def dump_pairs(pairs: list[KvPair]) -> str:
    return "\n".join(f"{item.key}={item.value}" for item in pairs)


def _prepare(dialog: QDialog) -> None:
    dialog.setWindowIcon(app_icon())
    dialog.setStyleSheet(APP_QSS)
    apply_dark_surface(dialog)


def _ok_cancel(dialog: QDialog) -> QDialogButtonBox:
    buttons = QDialogButtonBox()
    ok = buttons.addButton(t("ok"), QDialogButtonBox.ButtonRole.AcceptRole)
    cancel = buttons.addButton(t("cancel"), QDialogButtonBox.ButtonRole.RejectRole)
    ok.setObjectName("Primary")
    ok.clicked.connect(dialog.accept)
    cancel.clicked.connect(dialog.reject)
    return buttons


class ChannelDialog(QDialog):
    def __init__(self, parent: QWidget | None, kind: str, channel: MessageChannel | None = None) -> None:
        super().__init__(parent)
        self._kind = kind if kind in CHANNEL_KINDS else "webhook"
        self._id = channel.id if channel else new_id()
        self.setWindowTitle(t("channel.edit") if channel else t("channel.add"))
        _prepare(self)
        self.resize(560, 620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        hint = QLabel(t(f"channel.hint.{self._kind}"), objectName="Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        form.setSpacing(10)
        self.name = QLineEdit(channel.name if channel else kind_label(self._kind))
        form.addRow(t("channel.name"), self.name)
        self.enabled = QCheckBox(t("channel.enabled"))
        self.enabled.setChecked(True if channel is None else channel.enabled)
        form.addRow("", self.enabled)

        self.webhook = QLineEdit(channel.webhook if channel else "")
        self.webhook.setPlaceholderText("https://oapi.dingtalk.com/robot/send?access_token=")
        self.secret = QLineEdit(channel.secret if channel else "")
        self.secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.keyword = QLineEdit(channel.keyword if channel else "")
        self.app_key = QLineEdit(channel.app_key if channel else "")
        self.app_secret = QLineEdit(channel.app_secret if channel else "")
        self.app_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.robot_code = QLineEdit(channel.robot_code if channel else "")
        self.conversation = QLineEdit(channel.conversation_id if channel else "")
        self.url = QLineEdit(channel.url if channel else "")
        self.method = QComboBox()
        self.method.addItem("POST", "POST")
        self.method.addItem("GET", "GET")
        if channel and channel.method == "GET":
            self.method.setCurrentIndex(1)
        self.headers = QPlainTextEdit()
        self.headers.setPlaceholderText("X-Token=secret")
        self.headers.setPlainText(dump_pairs(channel.headers) if channel else "")
        self.params = QPlainTextEdit()
        self.params.setPlaceholderText("source=facehide")
        self.params.setPlainText(dump_pairs(channel.params) if channel else "")

        self.auth_keyword = QRadioButton(auth_label("keyword"))
        self.auth_sign = QRadioButton(auth_label("sign"))
        self.auth_ip = QRadioButton(auth_label("ip"))
        mode = channel.auth_mode if channel else "sign"
        self.auth_keyword.setChecked(mode == "keyword")
        self.auth_sign.setChecked(mode == "sign")
        self.auth_ip.setChecked(mode == "ip")
        auth_group = QButtonGroup(self)
        auth_group.addButton(self.auth_keyword)
        auth_group.addButton(self.auth_sign)
        auth_group.addButton(self.auth_ip)
        auth_row = QHBoxLayout()
        auth_row.addWidget(self.auth_keyword)
        auth_row.addWidget(self.auth_sign)
        auth_row.addWidget(self.auth_ip)
        auth_row.addStretch(1)
        auth_host = QWidget()
        auth_host.setLayout(auth_row)

        if self._kind == "dingtalk_group":
            self.webhook.setPlaceholderText("https://oapi.dingtalk.com/robot/send?access_token=")
            form.addRow(t("channel.webhook"), self.webhook)
            form.addRow(t("channel.auth"), auth_host)
            form.addRow(t("channel.keyword"), self.keyword)
            form.addRow(t("channel.secret"), self.secret)
            self.auth_keyword.toggled.connect(self._sync_auth)
            self.auth_sign.toggled.connect(self._sync_auth)
            self.auth_ip.toggled.connect(self._sync_auth)
            self._sync_auth()
        elif self._kind == "dingtalk_app":
            form.addRow(t("channel.app_key"), self.app_key)
            form.addRow(t("channel.app_secret"), self.app_secret)
            form.addRow(t("channel.robot_code"), self.robot_code)
            form.addRow(t("channel.conversation"), self.conversation)
            form.addRow(t("channel.keyword"), self.keyword)
        elif self._kind == "feishu":
            self.webhook.setPlaceholderText("https://open.feishu.cn/open-apis/bot/v2/hook/")
            form.addRow(t("channel.webhook"), self.webhook)
            form.addRow(t("channel.secret"), self.secret)
            form.addRow(t("channel.keyword"), self.keyword)
        else:
            form.addRow(t("channel.url"), self.url)
            form.addRow(t("channel.method"), self.method)
            form.addRow(t("channel.headers"), self.headers)
            form.addRow(t("channel.params"), self.params)
            extra = QLabel(t("channel.webhook_vars"), objectName="Hint")
            extra.setWordWrap(True)
            extra.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow("", extra)

        layout.addLayout(form)
        layout.addStretch(1)
        layout.addWidget(_ok_cancel(self))

    def _sync_auth(self) -> None:
        self.keyword.setEnabled(self.auth_keyword.isChecked())
        self.secret.setEnabled(self.auth_sign.isChecked())

    def result_channel(self) -> MessageChannel:
        if self.auth_keyword.isChecked():
            auth = "keyword"
        elif self.auth_ip.isChecked():
            auth = "ip"
        else:
            auth = "sign"
        name = self.name.text().strip() or kind_label(self._kind)
        return MessageChannel(
            id=self._id,
            kind=self._kind,
            name=name,
            enabled=self.enabled.isChecked(),
            webhook=self.webhook.text().strip(),
            secret=self.secret.text(),
            keyword=self.keyword.text().strip(),
            auth_mode=auth,
            app_key=self.app_key.text().strip(),
            app_secret=self.app_secret.text(),
            robot_code=self.robot_code.text().strip(),
            conversation_id=self.conversation.text().strip(),
            url=self.url.text().strip(),
            method=str(self.method.currentData() or "POST"),
            headers=parse_pairs(self.headers.toPlainText()),
            params=parse_pairs(self.params.toPlainText()),
        )
