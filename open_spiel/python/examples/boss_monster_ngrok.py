# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs the Boss Monster server (see `boss_monster_server.py`) behind an
ngrok tunnel, so two players anywhere can each open a private link.

Usage (locally or in Colab -- see `boss_monster_colab.ipynb` for a notebook
version of this exact flow):

    python3 open_spiel/python/examples/boss_monster_ngrok.py \
        --ngrok_authtoken=<your token from https://dashboard.ngrok.com>

Then open the printed public URL, click "New Game", and send the two
`/play/<token>` links it gives you to your two players.

Requires: `pip install flask pyngrok`.
"""

import time

from absl import app
from absl import flags
from pyngrok import ngrok

from open_spiel.python.examples import boss_monster_server as server

FLAGS = flags.FLAGS
flags.DEFINE_integer("local_port", 8080, "Local port to serve on.")
flags.DEFINE_string(
    "ngrok_authtoken", None,
    "Your ngrok authtoken (from https://dashboard.ngrok.com/get-started/"
    "your-authtoken). Required the first time you use ngrok on a machine.")
flags.DEFINE_string(
    "ngrok_domain", None,
    "Optional: a reserved ngrok domain, if you have one, for a stable URL "
    "across restarts.")


def run(port, ngrok_authtoken=None, ngrok_domain=None):
  """Starts the Flask app locally and opens an ngrok tunnel to it.

  Returns the `pyngrok` tunnel object; call `ngrok.disconnect(tunnel.public_url)`
  or `ngrok.kill()` to tear it down.
  """
  if ngrok_authtoken:
    ngrok.set_auth_token(ngrok_authtoken)

  import threading  # pylint: disable=g-import-not-at-top
  thread = threading.Thread(
      target=lambda: server.flask_app.run(  # pylint: disable=g-long-lambda
          host="0.0.0.0", port=port, threaded=True, use_reloader=False),
      daemon=True)
  thread.start()
  time.sleep(1.0)  # Give Flask a moment to bind the port.

  connect_kwargs = {}
  if ngrok_domain:
    connect_kwargs["domain"] = ngrok_domain
  tunnel = ngrok.connect(port, "http", **connect_kwargs)
  print("=" * 70)
  print(f"Boss Monster server is live at: {tunnel.public_url}")
  print("Open that URL, click 'New Game', and send the two /play/<token>")
  print("links it gives you to your two players.")
  print("=" * 70)
  return tunnel


def main(unused_argv):
  run(FLAGS.local_port, FLAGS.ngrok_authtoken, FLAGS.ngrok_domain)
  print("Press Ctrl+C to stop.")
  try:
    while True:
      time.sleep(3600)
  except KeyboardInterrupt:
    ngrok.kill()


if __name__ == "__main__":
  app.run(main)
