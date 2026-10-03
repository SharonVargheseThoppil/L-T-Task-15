import os

import requests
import streamlit as st


st.set_page_config(
    page_title="CIFAR-10 Deep Learning",
    page_icon="🤖",
    layout="centered"
)


API_URL = os.environ.get(
    "API_URL",
    "http://127.0.0.1:5000"
)


st.title("🤖 CIFAR-10 Image Classification")

st.write(
    "Upload an image and the deep learning model "
    "will classify it using the Flask API."
)

st.info(f"Backend API: {API_URL}")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔮 Predict"):

        try:

            files = {
                "image": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            with st.spinner("Sending image to Flask API..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=30
                )

            if response.status_code == 200:

                result = response.json()

                st.success("Prediction completed successfully.")

                st.subheader(
                    f"Prediction: {result['prediction']}"
                )

                st.metric(
                    "Confidence",
                    f"{result['confidence_percentage']}%"
                )

                st.write(
                    f"API Version: {result['version']}"
                )

                st.subheader("Class Probabilities")

                probabilities = result.get(
                    "probabilities",
                    {}
                )

                st.bar_chart(probabilities)

            else:

                try:
                    error_data = response.json()
                    st.error(
                        error_data.get(
                            "message",
                            "Prediction request failed."
                        )
                    )

                except Exception:
                    st.error(
                        f"API returned status code "
                        f"{response.status_code}"
                    )

        except requests.exceptions.Timeout:

            st.error(
                "The Flask API request timed out."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the Flask API."
            )

        except Exception as error:

            st.error(
                f"Unexpected error: {error}"
            )


st.divider()

st.subheader("API Health")

if st.button("Check API Health"):

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=10
        )

        if response.status_code == 200:

            data = response.json()

            st.success(
                f"API is healthy — Version: "
                f"{data.get('version', 'unknown')}"
            )

        else:

            st.error(
                f"API returned status "
                f"{response.status_code}"
            )

    except Exception as error:

        st.error(
            f"API unavailable: {error}"
        )