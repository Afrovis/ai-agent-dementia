# listen

`listen` drains `audio_in` continuously but runs VAD/STT only while a session
is `OBSERVING` or later, so silence from it outside a session is expected. Its
first complete utterance downloads `small.en` into
`data/models/faster-whisper`; later container recreations reuse those weights.
The service tests inject both VAD and transcription, so they need no hardware
or model download.

Barge-in does not wait for Whisper. `listen` waits for sustained WebRTC VAD
speech and confirms the window with the bundled Silero model before publishing
`SpeechStarted` on `speech_in`. While the page is speaking, playback activity
raises the threshold. There is no software echo cancellation; the page relies
on the browser/OS echo cancellation it requests from `getUserMedia`.
