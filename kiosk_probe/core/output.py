from rich.console import Console


class UserOutput:

    def __init__(self):
        self.console = Console()

    def report(self, content: str):
        self.console.print(f"\r{content}")

    def report_note(self, content: str):
        self.console.print(f"\r[dim]{content}[/]")

    def report_warning(self, content: str):
        self.console.print(f"\r⚠️ [orange1 italic]{content}[/]")

    def report_error(self, content: str):
        self.console.print(f"\r❌ [bright_red bold]{content}[/]")

    def report_success(self, content: str):
        self.console.print(f"\r✅ [green4]{content}[/]")

    @staticmethod
    def clear_transient():
        print(end="\r", flush=True)
