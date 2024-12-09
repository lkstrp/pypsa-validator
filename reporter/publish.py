import gitlab
import re
from reporter.utils import REPORT_IDENTIFIER, read_env_var
from reporter.meta.paths import PATH_ARTIFACTS
from reporter.meta.config import PREFIX, PLATFORM, DOMAIN


def publish(text: str) -> None:
    """Publish report."""
    if PLATFORM == "gitlab":
        publish_on_gitlab(text)


def publish_on_gitlab(text: str) -> None:
    """Publish report on GitLab."""
    GL_TOKEN = read_env_var("GITLAB_TOKEN")
    PROJECT_ID = read_env_var("CI_PROJECT_ID")
    MR_IID = read_env_var("CI_MERGE_REQUEST_IID")

    gl = gitlab.Gitlab(DOMAIN, private_token=GL_TOKEN)
    project = gl.projects.get(PROJECT_ID)
    mr = project.mergerequests.get(MR_IID)

    def _upload_image_to_github(name):
        path = PATH_ARTIFACTS / name
        assert path.exists(), f"File {path} does not exist"
        uploaded_file = project.upload(name, filedata=open(path, "rb"))
        return uploaded_file["url"]

    # Upload placeholder images
    text = _replace_placeholders(text, _upload_image_to_github)

    existing = next(
        (note for note in mr.notes.list() if REPORT_IDENTIFIER in note.body), None
    )

    if existing:
        existing.body = text
        existing.save()
    else:
        mr.notes.create({"body": text})


def _replace_placeholders(s, func):
    return re.sub(r"<URL_IMAGE_PLACEHOLDER:(.*?)>", lambda m: func(m.group(1)), s)
