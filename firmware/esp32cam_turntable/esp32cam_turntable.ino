// ESP32-CAM (AI-Thinker) + three standard (positional, 180°) servos:
// one turns the turntable, one tilts the camera, one pans the camera left/right.
// The ESP32 stores nothing: your computer asks for each photo over Wi-Fi.
//   http://<board-ip>/                        test page (photo, live view, move the servos)
//   http://<board-ip>/capture                 one photo as a JPEG image (not while streaming)
//   http://<board-ip>/angle?deg=90            move the TURN servo to 90° (0..180)
//   http://<board-ip>/angle?deg=90&servo=tilt move the TILT servo to 90° (0..180)
//   http://<board-ip>/angle?deg=90&servo=pan  move the PAN servo to 90° (0..180)
//   http://<board-ip>/angle?by=-10&servo=pan  move a servo 10° from where it is now
//   http://<board-ip>/pulse?us=1500[&servo=tilt|pan]  raw pulse width, to find MIN_US / MAX_US
//   http://<board-ip>:81/stream?size=qvga     live MJPEG video for live_detect.py
//                                             size: qqvga 160x120, qvga 320x240 (default),
//                                             hvga 480x320, vga 640x480, svga 800x600
//                                             optional &q=12 (JPEG quality 8..63, lower = better)
//   The stream runs on its own port (81), so the servos can be moved while it is open.
//   http://<board-ip>/stream?... redirects there. /capture waits until the stream is closed.
//
// Arduino IDE: board "AI Thinker ESP32-CAM" (this enables the PSRAM the camera needs).

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// The board also answers at http://heronet-cam.local/ (no need to know its IP address).
const char* HOSTNAME = "heronet-cam";

const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASS = "YOUR_PASSWORD";

// ---- Photo size and quality ----
// FRAMESIZE_UXGA = 1600x1200 (max), FRAMESIZE_SXGA = 1280x1024, FRAMESIZE_XGA = 1024x768,
// FRAMESIZE_SVGA = 800x600, FRAMESIZE_VGA = 640x480
#define FRAME_SIZE    FRAMESIZE_UXGA
#define JPEG_QUALITY  10     // 4..63, lower = better quality and bigger files

// ---- Wi-Fi ----
// false = power saving off: fast transfers. If the board keeps restarting
// ("Brownout detector was triggered"), use a stronger 5 V supply or set this to true.
#define WIFI_POWER_SAVE false

// ---- Servos: standard 180° servos (SG90, MG90S, MG996R...) ----
// Power them from their own 5-6 V supply; connect the supply's - to an ESP32-CAM GND pin.
// Pulse widths for 0° and 180°: typical values are 500 and 2400-2500 µs, but each
// servo differs. Use /pulse?us=... to find where yours really is at 0° and 180°.
#define SERVO_FREQ    50     // standard servo: one pulse every 20 ms
#define SERVO_BITS    14     // PWM resolution
#define SERVO_DEG_PER_SEC 60 // movement speed: slow = less wobble

struct ServoCfg {
  const char* name;
  uint8_t  pin;
  uint16_t minUs, maxUs;     // pulse width at 0° and at 180°
  float    deg;              // current angle; -1 = unknown (after boot)
};
// GPIO 13, 14 and 15 are free on the ESP32-CAM when no SD card is used.
ServoCfg turnServo = { "turn", 14, 500, 2400, -1 };   // turntable
ServoCfg tiltServo = { "tilt", 15, 500, 2400, -1 };   // tilt (camera up/down)
ServoCfg panServo  = { "pan",  13, 500, 2400, -1 };   // pan (camera left/right)

// ---- AI-Thinker ESP32-CAM camera pins (fixed on the board) ----
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22
#define FLASH_LED_PIN   4    // the bright white LED; kept off

WebServer server(80);         // photos, servos, test page
WebServer streamServer(81);   // live video, handled by its own task
volatile bool streaming = false;

bool initCamera() {
  camera_config_t c = {};
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0;       // the servos use a different PWM group, no conflict
  c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_JPEG;   // the camera compresses to JPEG itself
  c.frame_size = FRAME_SIZE;
  c.jpeg_quality = JPEG_QUALITY;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;  // always hand out the newest frame

  if (!psramFound()) {
    Serial.println("No PSRAM found: select the board 'AI Thinker ESP32-CAM' in Arduino IDE.");
    c.frame_size = FRAMESIZE_SVGA;
    c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  Serial.printf("Camera OK, sensor PID: 0x%x\n", s->id.PID);
  // s->set_vflip(s, 1);    // uncomment if the image is upside down
  // s->set_hmirror(s, 1);  // uncomment if the image is mirrored
  return true;
}

void handleCapture() {
  for (int i = 0; streaming && i < 150; i++) delay(20);   // give a closing stream up to 3 s
  if (streaming) {
    server.send(409, "text/plain", "A live stream is open: close it before taking photos.");
    return;
  }
  uint32_t t0 = millis();
  // Throw away one frame so the photo shows the scene as it is now (after a servo move).
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb) esp_camera_fb_return(fb);
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed");
    server.send(500, "text/plain", "Capture failed");
    return;
  }
  uint32_t tCap = millis() - t0;

  server.sendHeader("Cache-Control", "no-store");
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.sendContent((const char*)fb->buf, fb->len);
  Serial.printf("Photo: %ux%u, %u KB, %.1f s capturing + %.1f s sending\n",
                fb->width, fb->height, fb->len / 1024,
                tCap / 1000.0, (millis() - t0 - tCap) / 1000.0);
  esp_camera_fb_return(fb);
}

// ---- Live MJPEG stream ----
#define STREAM_QUALITY 12    // default JPEG quality for the stream

framesize_t streamSize(const String& name) {
  if (name == "qqvga") return FRAMESIZE_QQVGA;
  if (name == "hvga")  return FRAMESIZE_HVGA;
  if (name == "vga")   return FRAMESIZE_VGA;
  if (name == "svga")  return FRAMESIZE_SVGA;
  return FRAMESIZE_QVGA;
}

void dropFrames(int n) {              // after a size change the first frames can be stale
  for (int i = 0; i < n; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
}

void handleStream() {
  if (streaming) {                    // one viewer at a time
    streamServer.send(409, "text/plain", "Another stream is already open.");
    return;
  }
  streaming = true;
  sensor_t* s = esp_camera_sensor_get();
  String sizeName = streamServer.hasArg("size") ? streamServer.arg("size") : "qvga";
  int q = streamServer.hasArg("q") ? constrain(streamServer.arg("q").toInt(), 8, 63) : STREAM_QUALITY;
  s->set_framesize(s, streamSize(sizeName));
  s->set_quality(s, q);
  dropFrames(2);

  WiFiClient client = streamServer.client();
  client.setNoDelay(true);
  client.print("HTTP/1.1 200 OK\r\n"
               "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
               "Cache-Control: no-store\r\n"
               "Access-Control-Allow-Origin: *\r\n"
               "Connection: close\r\n\r\n");
  Serial.printf("Stream started (%s, quality %d)\n", sizeName.c_str(), q);

  uint32_t frames = 0, t0 = millis();
  while (client.connected()) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { delay(5); continue; }
    char head[96];
    int n = snprintf(head, sizeof(head),
                     "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    bool ok = client.write((const uint8_t*)head, n) == (size_t)n &&
              client.write(fb->buf, fb->len) == fb->len &&
              client.write((const uint8_t*)"\r\n", 2) == 2;
    uint16_t w = fb->width, h = fb->height;
    esp_camera_fb_return(fb);
    if (!ok) break;
    frames++;
    if (millis() - t0 >= 5000) {
      Serial.printf("  stream %ux%u: %.1f fps\n", w, h, frames * 1000.0 / (millis() - t0));
      frames = 0;
      t0 = millis();
    }
  }
  client.stop();
  s->set_framesize(s, FRAME_SIZE);    // back to full size for /capture
  s->set_quality(s, JPEG_QUALITY);
  dropFrames(2);
  streaming = false;
  Serial.println("Stream stopped");
}

// The old address http://<ip>/stream?... still works: it sends the browser to port 81.
void handleStreamRedirect() {
  String url = "http://" + WiFi.localIP().toString() + ":81/stream";
  String args;
  for (int i = 0; i < server.args(); i++)
    args += String(i ? "&" : "?") + server.argName(i) + "=" + server.arg(i);
  server.sendHeader("Location", url + args);
  server.send(302, "text/plain", "Moved to port 81");
}

void streamTask(void*) {
  for (;;) {
    streamServer.handleClient();
    delay(2);
  }
}

// ---- Servo helpers ----
void servoPulseUs(ServoCfg& s, uint16_t us) {
  // duty = pulse / period, at 14-bit resolution
  ledcWrite(s.pin, (uint32_t)us * ((1 << SERVO_BITS) - 1) / (1000000 / SERVO_FREQ));
}

void servoWriteDeg(ServoCfg& s, float deg) {
  servoPulseUs(s, s.minUs + (s.maxUs - s.minUs) * deg / 180.0f);
}

// Moves smoothly to `target` degrees, then stops sending pulses so the servo
// doesn't buzz while the photo is taken.
void moveServo(ServoCfg& s, float target) {
  target = constrain(target, 0.0f, 180.0f);
  if (s.deg < 0) {                    // position unknown after boot: go there directly
    servoWriteDeg(s, target);
    delay(1000);
  } else {
    const float stepDeg = SERVO_DEG_PER_SEC / (float)SERVO_FREQ;   // per 20 ms pulse
    float pos = s.deg;
    while (fabsf(target - pos) > stepDeg) {
      pos += (target > pos) ? stepDeg : -stepDeg;
      servoWriteDeg(s, pos);
      delay(1000 / SERVO_FREQ);
    }
    servoWriteDeg(s, target);
    delay(400);                       // let it settle on the exact angle
  }
  s.deg = target;
  ledcWrite(s.pin, 0);                // release: no pulses while idle
}

ServoCfg& servoFromArg() {
  String name = server.arg("servo");
  if (name == "tilt") return tiltServo;
  if (name == "pan") return panServo;
  return turnServo;
}

void handleAngle() {
  if (!server.hasArg("deg") && !server.hasArg("by")) {
    server.send(400, "text/plain", "use /angle?deg=0..180 or /angle?by=-10 [&servo=tilt|pan]");
    return;
  }
  ServoCfg& s = servoFromArg();
  float deg = server.hasArg("deg") ? server.arg("deg").toFloat()
                                   : (s.deg < 0 ? 90.0f : s.deg) + server.arg("by").toFloat();
  Serial.printf("Moving %s servo (GPIO %u) to %.1f deg...\n", s.name, s.pin, deg);
  moveServo(s, deg);
  Serial.printf("  done\n");
  server.send(200, "text/plain", String(s.name) + " angle = " + s.deg);
}

// Holds a raw pulse width for 2 s, so you can find a servo's real 0° and 180°.
void handlePulse() {
  ServoCfg& s = servoFromArg();
  uint16_t us = constrain(server.arg("us").toInt(), 400, 2600);
  Serial.printf("Sending %u us pulses to the %s servo (GPIO %u) for 2 s\n", us, s.name, s.pin);
  servoPulseUs(s, us);
  delay(2000);
  ledcWrite(s.pin, 0);
  s.deg = -1;                         // position no longer matches an angle we know
  server.send(200, "text/plain", String(s.name) + " pulse = " + us + " us");
}

void handleRoot() {
  String turn, tilt, pan;
  for (int a = 0; a <= 180; a += 45)
    turn += "<button onclick=\"fetch('/angle?deg=" + String(a) + "')\">" + String(a) + "&deg;</button> ";
  for (int a = 60; a <= 120; a += 10)
    tilt += "<button onclick=\"fetch('/angle?servo=tilt&deg=" + String(a) + "')\">" + String(a) + "&deg;</button> ";
  pan = "<button onclick=\"fetch('/angle?servo=pan&by=-10')\">&larr; 10&deg;</button> ";
  for (int a = 0; a <= 180; a += 30)
    pan += "<button onclick=\"fetch('/angle?servo=pan&deg=" + String(a) + "')\">" + String(a) + "&deg;</button> ";
  pan += "<button onclick=\"fetch('/angle?servo=pan&by=10')\">10&deg; &rarr;</button>";
  String live = "http://" + WiFi.localIP().toString() + ":81/stream";
  server.send(200, "text/html",
    "<!doctype html><html><head><meta name=viewport content='width=device-width'>"
    "<title>ESP32-CAM</title></head><body style='font-family:sans-serif;text-align:center'>"
    "<h2>ESP32-CAM turntable</h2>"
    "<p>Turn servo: " + turn + "</p><p>Tilt servo: " + tilt + "</p><p>Pan servo: " + pan + "</p>"
    "<button onclick=\"p.src='" + live + "?size=qvga'\" style='font-size:1.2em'>Live view</button> "
    "<button onclick=\"p.src='';setTimeout(()=>p.src='/capture?t='+Date.now(),300)\" "
    "style='font-size:1.2em'>Take photo</button><br><br>"
    "<img id=p style='width:95%;max-width:1024px'>"
    "<p><a href='/capture' download='photo.jpg'>Download a photo</a> &middot; "
    "<a href='" + live + "?size=vga' target=_blank>Live 640x480 in a new tab</a></p>"
    "<small>Taking a photo closes the live view first.</small>"
    "</body></html>");
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  if (!initCamera()) {
    Serial.println("Check that the camera ribbon cable is fully inserted and latched.");
    while (true) delay(1000);
  }

  for (ServoCfg* s : { &turnServo, &tiltServo, &panServo }) {
    if (ledcAttach(s->pin, SERVO_FREQ, SERVO_BITS)) {
      Serial.printf("Servo '%s' ready on GPIO %u\n", s->name, s->pin);
    } else {
      Serial.printf("PWM setup failed for the %s servo\n", s->name);
    }
    ledcWrite(s->pin, 0);        // no pulses until the first /angle command
  }

  WiFi.mode(WIFI_STA);
  WiFi.setHostname(HOSTNAME);
  WiFi.setSleep(WIFI_POWER_SAVE);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\nOpen http://%s/\n", WiFi.localIP().toString().c_str());
  if (MDNS.begin(HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("      or http://%s.local/\n", HOSTNAME);
  }

  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.on("/angle", handleAngle);
  server.on("/pulse", handlePulse);
  server.on("/stream", handleStreamRedirect);
  server.begin();

  streamServer.on("/stream", handleStream);
  streamServer.begin();
  xTaskCreatePinnedToCore(streamTask, "stream", 8192, NULL, 1, NULL, 0);
  Serial.printf("Live stream: http://%s:81/stream\n", WiFi.localIP().toString().c_str());
}

void loop() {
  server.handleClient();
}
