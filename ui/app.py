import asyncio
import pandas as pd
import streamlit as st
from markdown import markdown
import altair as alt
from ui.config import API_ENDPOINT
from ui.utils import (
    server_is_ready,
    upload_audio_file,
    process_audio,
    get_audio_info,
    get_speech_analysis,
)


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def stop(task, status_placeholder):
    task.cancel()
    st.session_state.server_status_label = str(f"Service Status: ERROR")
    with status_placeholder.container():
        st.write(
            markdown(st.session_state.server_status_label),
            unsafe_allow_html=True,
        )


async def check_server_status(status_placeholder):
    while True:
        print("Checking application status")
        _ , response_is_ready = server_is_ready()
        st.session_state.server_status_label = str(f"Service Status : {response_is_ready['applications']['ASR']['status']}")
        with status_placeholder.container():
            st.write(
                markdown(st.session_state.server_status_label),
                unsafe_allow_html=True,
            )
        if response_is_ready["applications"]["ASR"]["status"] == "RUNNING":
            st.session_state.server_current_status = "RUNNING"
            break
        await asyncio.sleep(1)


@st.cache_data
def upload_audio(input_audio_file):
    response = upload_audio_file(input_audio_file)
    return response


def chart_speaker_talk_time(df_spk_dist):
    bars = alt.Chart(df_spk_dist).mark_bar().encode(x="talktime:Q", y="speaker:O")

    text = bars.mark_text(
        align="left",
        baseline="middle",
        dx=3,  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(text="talktime:Q")

    chart_talk_time = (bars + text).properties(height=300)

    chart_talk_perc = (
        alt.Chart(df_spk_dist)
        .transform_joinaggregate(
            TotalTime="sum(talktime)",
        )
        .transform_calculate(talk_time="datum.talktime / datum.TotalTime")
        .mark_bar()
        .encode(alt.X("talk_time:Q", axis=alt.Axis(format=".0%")), y="speaker:N")
    )

    text_2 = chart_talk_perc.mark_text(
        align="left",
        baseline="middle",
        dx=3,  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(text="talk_time:Q")
    chart_talk_perc = (chart_talk_perc + text_2).properties(height=300)
    tab1, tab2 = st.tabs(["Talk Time (s)", "Talk Time (%)"])
    with tab1:
        st.altair_chart(chart_talk_time, theme="streamlit", use_container_width=True)
    with tab2:
        st.altair_chart(chart_talk_perc, theme="streamlit", use_container_width=True)


def show_speech_analysis(speech_analysis_placeholder):

    with speech_analysis_placeholder.container():
        st.write("### Speaker Talk Time Distribution")
        ttd = st.session_state.current_speech_analysis_results[
            "speaker_talk_time_distribution"
        ]
        df_spk_dist = pd.DataFrame(ttd, index=[0])
        df_spk_dist = df_spk_dist.T.reset_index()
        df_spk_dist.columns = ["speaker", "talktime"]
        chart_speaker_talk_time(df_spk_dist)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Speaker Words/Minute")
            st.divider()
            wpm = st.session_state.current_speech_analysis_results["words_per_min"]
            df_wpm = pd.DataFrame(wpm, index=[0])
            df_wpm = df_wpm.T.reset_index()
            df_wpm.columns = ["speaker", "Words/Minute"]
            st.dataframe(df_wpm)
        with col2:
            st.write("### Longest Monologue")
            st.divider()
            wpm = st.session_state.current_speech_analysis_results["longest_monologue"]
            df_mono = pd.DataFrame(wpm, index=[0])
            df_mono.columns = ["Monologue Time", "Speaker", "Text", "Start", "End"]
            st.dataframe(df_mono)
            st.write(
                f"### Switches in conversation : {st.session_state.current_speech_analysis_results['switches_per_conversation']}"
            )


def main():  # pylint: disable=R0914,R0912,R0915
    st.set_page_config(page_title="ASR Demo", page_icon="")
    set_state_if_absent(
        "server_status_label",
        markdown(f"Service Status: Unavailable"),
    )
    set_state_if_absent("server_current_status", "Unavailable")
    set_state_if_absent("current_audio_file_on_server", None)
    set_state_if_absent("current_transcription_results", None)
    set_state_if_absent("current_speech_analysis_results", None)
    set_state_if_absent("current_audio_info", None)

    # with open("fastrag/ui/style.css", "r") as f:
    # st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.write("# Speech-to-Text Analysis")

    with st.sidebar:
        server_status_placeholder = st.empty()
        with server_status_placeholder.container():
            st.write(
                markdown(st.session_state.server_status_label),
                unsafe_allow_html=True,
            )

    input_audio_file = st.sidebar.file_uploader(
        ":blue[**Upload an audio file:**]",
        type=[".wav", ".mp3", ".flac"],
        accept_multiple_files=False,
    )
    # st.sidebar.header("Options")
    process_audio_btn = None
    if input_audio_file:
        response = upload_audio(input_audio_file)
        if response is not None:
            st.session_state.current_audio_file_on_server = response["file_url"]
            print(f"Uploaded file , server file url {response['file_url']}")
            bytes_data = input_audio_file.getvalue()

            st.audio(bytes_data)
            st.divider()
            st.sidebar.divider()

            col1, col2 = st.sidebar.columns(2)
            with col1:
                is_alignment = st.checkbox("Alignment")
            with col2:
                is_speaker_aware = st.checkbox("Speaker Aware")
                is_speech_analysis = None
                if is_speaker_aware:
                    is_speech_analysis = st.checkbox("Speech Analysis")
                    st.session_state.current_audio_info = get_audio_info(
                        input_audio_file
                    )
                    print(st.session_state.current_audio_info)
            process_audio_btn = st.sidebar.button("Process Audio")
        else:
            st.sidebar.error("Error in uploading file.")

    # Check the connection
    if st.session_state.server_current_status != "RUNNING":
        with st.spinner("⌛️ &nbsp;&nbsp; ASR service is starting..."):
            is_ready, response_is_ready = server_is_ready()
            print(f"Service is ready {is_ready} Response: {response_is_ready}")
            if not is_ready:
                st.error(
                    "🚫 &nbsp;&nbsp; Connection Error. \
                    Is the ASR pipeline service running?"
                )
                st.error(f"Using endpoint: {API_ENDPOINT}")
                st.session_state.server_status_label = str(f"Service Status: Unavailable")
            else:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                task = loop.create_task(check_server_status(server_status_placeholder))
                loop.call_later(30, stop, task, server_status_placeholder)
                try:
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    pass

            # reset_results()

    transcription_result_placeholder = st.empty()
    speech_analysis_placeholder = st.empty()

    if st.session_state.current_transcription_results is not None:
        with transcription_result_placeholder.container():
            st.write("### Transcription")
            st.dataframe(st.session_state.current_transcription_results)

    if st.session_state.current_speech_analysis_results is not None:
        show_speech_analysis(speech_analysis_placeholder)

    if process_audio_btn:
        with st.spinner("⌛️ &nbsp;&nbsp; Processing audio..."):
            transcription_result_placeholder.empty()
            speech_analysis_placeholder.empty()

            response = process_audio(
                st.session_state.current_audio_file_on_server,
                is_alignment,
                is_speaker_aware,
            )
            if response is not None and "segments" in response.keys():
                df = pd.DataFrame.from_dict(response["segments"])
                st.session_state.current_transcription_results = df
                with transcription_result_placeholder.container():
                    st.write("### Transcription")
                    st.dataframe(st.session_state.current_transcription_results)
                if is_speech_analysis:
                    speech_analysis_response = get_speech_analysis(
                        response, st.session_state.current_audio_info["duration"]
                    )
                    if speech_analysis_response is not None:
                        st.session_state.current_speech_analysis_results = (
                            speech_analysis_response
                        )
                        show_speech_analysis(speech_analysis_placeholder)
            else:
                st.error("Error in processing an audio. Try again")


main()
