"""Endpoints page: the LLM APIs runs can call, and their keys."""

from __future__ import annotations

import httpx
import streamlit as st

from experiments.runs.endpoints import (
    OPENROUTER_KEY_ENV, delete_endpoint, endpoints_path, key_source, list_endpoints, save_endpoint,
    set_openrouter_key,
)
from experiments.runs.llm import list_endpoint_models

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


def test_connection(base_url: str | None, api_key: str | None) -> None:
    """Show whether the endpoint answers with this key."""
    try:
        if base_url is None:
            resp = httpx.get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            resp.raise_for_status()
            label = (resp.json().get("data") or {}).get("label") or "key"
            st.success(f"OpenRouter accepts this key ({label}).", icon=":material/check_circle:")
        else:
            models = list_endpoint_models(base_url, api_key)
            st.success(f"{len(models)} model(s): {', '.join(models[:8])}{' ...' if len(models) > 8 else ''}",
                       icon=":material/check_circle:")
    except Exception as exc:  # noqa: BLE001 - unreachable, refused key, no /models route...
        st.error(f"{type(exc).__name__}: {exc}", icon=":material/error:")


st.title("Endpoints")
st.caption(
    f"The LLM APIs runs can call. Keys are saved in `{endpoints_path().name}` under `secrets/` (gitignored, readable "
    "by your user only) and never in a run's settings, the run history or its output folder: a run records the "
    "endpoint's URL and the worker looks the key up when the run starts. Keys are never shown again once saved."
)

endpoints = list_endpoints()
openrouter, custom = endpoints[0], endpoints[1:]

with st.container(border=True):
    st.markdown("**OpenRouter**")
    st.caption(f":material/key: {key_source(openrouter)}"
               + ("" if openrouter.api_key else f" · a saved key takes precedence over `{OPENROUTER_KEY_ENV}`"))
    with st.form("openrouter", border=False):
        key = st.text_input("API key", type="password", placeholder="sk-or-v1-...")
        c = st.container(horizontal=True)
        save = c.form_submit_button("Save key", type="primary")
        tested = c.form_submit_button("Test")
        forget = c.form_submit_button("Forget saved key", disabled=not openrouter.api_key)
    if save:
        if key.strip():
            set_openrouter_key(key)
            st.toast("OpenRouter key saved.", icon=":material/check:")
            st.rerun()
        st.warning("Paste a key first.")
    if tested:
        test_connection(None, key.strip() or openrouter.api_key or None)
    if forget:
        set_openrouter_key(None)
        st.rerun()

st.subheader("Custom endpoints")
st.caption(
    "Any server speaking the OpenAI chat completions API: on-premise vLLM, Ollama, TGI, LM Studio... "
    "The base URL goes up to and including `/v1`. Runs execute in the worker container, so a server on this "
    "machine is `http://host.docker.internal:<port>/v1`, not `localhost`."
)
for e in custom:
    with st.expander(f"**{e.name}** · `{e.base_url}` · {key_source(e)}", icon=":material/dns:"):
        with st.form(f"edit_{e.name}", border=False):
            c1, c2 = st.columns(2)
            name = c1.text_input("Name", value=e.name)
            url = c2.text_input("Base URL", value=e.base_url)
            key = st.text_input("API key", type="password", placeholder="leave empty to keep the saved key"
                                if e.api_key else "none - leave empty if the server needs no key")
            drop_key = st.checkbox("Remove the saved key", disabled=not e.api_key)
            c = st.container(horizontal=True)
            save = c.form_submit_button("Save", type="primary")
            tested = c.form_submit_button("Test")
            delete = c.form_submit_button("Delete")
        if save:
            try:
                save_endpoint(name, url, None if drop_key else key, keep_key=not drop_key, old_name=e.name)
                st.toast(f"Saved {name}.", icon=":material/check:")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if tested:
            test_connection(url, key.strip() or (None if drop_key else e.api_key))
        if delete:
            delete_endpoint(e.name)
            st.rerun()
if not custom:
    st.info("No custom endpoint yet.", icon=":material/dns:")

with st.container(border=True):
    st.markdown("**Add an endpoint**")
    # A new form key after each add empties the fields (clear_on_submit would also clear them on "Test").
    with st.form(f"add_{st.session_state.get('added', 0)}", border=False):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name", placeholder="SINTEF vLLM")
        url = c2.text_input("Base URL", placeholder="http://gpu01:8000/v1")
        key = st.text_input("API key (optional)", type="password", placeholder="leave empty if the server needs no key")
        c = st.container(horizontal=True)
        add = c.form_submit_button("Add", type="primary", icon=":material/add:")
        tested = c.form_submit_button("Test")
    if add:
        try:
            save_endpoint(name, url, key, keep_key=False)
            st.session_state["added"] = st.session_state.get("added", 0) + 1
            st.toast(f"Added {name.strip()}.", icon=":material/check:")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if tested and url.strip():
        test_connection(url, key.strip() or None)
