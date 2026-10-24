"""Optional local browser UI: film-agent-web."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .core import STYLES, load_photo, process_array


def main() -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Для интерфейса установите: pip install -e '.[web]'") from exc

    def process(photo: str, style: str, strength: float, grain: float, auto: bool):
        if not photo:
            raise gr.Error("Загрузите фотографию")
        pixels, meta = load_photo(photo)
        out, report = process_array(pixels, style=style, strength=strength,
                                    grain=grain, iso=meta["iso"], auto=auto)
        return Image.fromarray(out), json.dumps(report, ensure_ascii=False, indent=2)

    with gr.Blocks(title="Film Agent · Локальная фотолаборатория") as app:
        gr.Markdown("# Film Agent\nАдаптивная ч/б среднеформатная и цветная Kodak 35 мм обработка. Фото остаются на вашем компьютере.")
        with gr.Row():
            photo = gr.Image(label="Исходник", type="filepath", sources=["upload"])
            result = gr.Image(label="Результат", format="png")
        style = gr.Dropdown(choices=[(v.name, k) for k, v in STYLES.items()],
                            value="hp5-120", label="Плёнка")
        with gr.Row():
            strength = gr.Slider(0, 2, value=1, step=0.05, label="Характер плёнки")
            grain = gr.Slider(0, 3, value=1, step=0.05, label="Зерно")
        auto = gr.Checkbox(label="Адаптировать экспозицию, WB и шумы", value=True)
        run = gr.Button("Проявить", variant="primary")
        details = gr.Code(label="Отчёт о корректировках", language="json")
        run.click(process, inputs=[photo, style, strength, grain, auto], outputs=[result, details])
    app.launch(inbrowser=True, share=False)


if __name__ == "__main__":
    main()
