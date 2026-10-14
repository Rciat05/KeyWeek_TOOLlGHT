/*
  detector_focos.ino
  Recibe por Bluetooth (HC-05 / HC-06) líneas de texto con dígitos y enciende los focos
  con esos números. Compatible con Arduino Uno / Nano.

  PROTOCOLO (una línea por mensaje, terminada en '\n'):
    "135\n" -> enciende los focos 1, 3 y 5 y apaga los demás
    "4\n"   -> enciende solo el foco 4
    "0\n"   -> apaga todos
  SEGURIDAD: si pasan TIEMPO_SIN_SENAL_MS sin recibir ningún mensaje (se cerró el
  programa, se perdió la conexión...), se apagan todos los focos.

  CONEXIONES
    Módulo Bluetooth:
      VCC -> 5V        GND -> GND
      TXD del módulo -> pin 10 del Arduino (RX por software)
      RXD del módulo -> pin 11 del Arduino, A TRAVÉS DE UN DIVISOR DE TENSIÓN
                        (el módulo trabaja a 3,3 V en sus entradas):
                        pin 11 --[1 kΩ]--+--[2 kΩ]-- GND   y   RXD del módulo en el punto medio (+)
    Focos 1 a 8 -> pines 2 a 9 (ver PINES_FOCOS).
      * Con LEDs: pin -> resistencia de 220 Ω -> LED -> GND.
      * Con focos de red eléctrica (110/220 V): NUNCA se conectan directo a los pines.
        Usa un módulo de relés de 8 canales (o relés/SSR adecuados) y respeta las
        normas de seguridad eléctrica: caja cerrada, cables de sección adecuada, sin
        partes con tensión al descubierto. Si no tienes experiencia con tensión de red,
        pide ayuda o usa LEDs / focos de baja tensión.
      * Muchos módulos de relés se activan con nivel BAJO: pon ACTIVO_EN_ALTO = false.

  El módulo emparejado con Windows aparece como un puerto COM (PIN habitual: 1234 o 0000).
  Velocidad del módulo en modo datos: 9600 baudios por defecto (HC-06 y HC-05).
*/

#include <SoftwareSerial.h>

// ---------------- Configuración ----------------
const uint8_t PIN_BT_RX = 10;   // recibe del TXD del módulo
const uint8_t PIN_BT_TX = 11;   // envía al RXD del módulo (con divisor de tensión)
const long BAUDIOS_BT = 9600;

const uint8_t NUM_FOCOS = 8;
const uint8_t PINES_FOCOS[NUM_FOCOS] = {2, 3, 4, 5, 6, 7, 8, 9};  // foco 1 -> pin 2, ... foco 8 -> pin 9
const bool ACTIVO_EN_ALTO = true;                // false para módulos de relés que activan con LOW

const unsigned long TIEMPO_SIN_SENAL_MS = 1500;  // apaga todo si no llega ningún mensaje
// -----------------------------------------------

SoftwareSerial bt(PIN_BT_RX, PIN_BT_TX);

char linea[24];
uint8_t largo = 0;
unsigned long ultimoMensaje = 0;
uint8_t mascaraActual = 0;   // bit 0 = foco 1, ..., bit 7 = foco 8

void escribirFoco(uint8_t i, bool encendido) {
  bool nivel = ACTIVO_EN_ALTO ? encendido : !encendido;
  digitalWrite(PINES_FOCOS[i], nivel ? HIGH : LOW);
}

void aplicarMascara(uint8_t mascara) {
  for (uint8_t i = 0; i < NUM_FOCOS; i++) {
    escribirFoco(i, mascara & (1 << i));
  }
  mascaraActual = mascara;
}

void procesarLinea() {
  uint8_t mascara = 0;
  bool hayDigitos = false;
  for (uint8_t i = 0; i < largo; i++) {
    char c = linea[i];
    if (c >= '0' && c <= '9') hayDigitos = true;
    if (c >= '1' && c <= '8') mascara |= (1 << (c - '1'));
  }
  if (hayDigitos) {                  // ignora líneas vacías o con ruido
    aplicarMascara(mascara);
    ultimoMensaje = millis();
    Serial.print(F("Mensaje: "));
    Serial.print(linea);
    Serial.print(F("  mascara: "));
    Serial.println(mascara, BIN);
  }
  largo = 0;
}

void setup() {
  Serial.begin(9600);                // monitor serie USB (solo depuración)
  bt.begin(BAUDIOS_BT);
  for (uint8_t i = 0; i < NUM_FOCOS; i++) {
    escribirFoco(i, false);          // primero el nivel "apagado"...
    pinMode(PINES_FOCOS[i], OUTPUT); // ...y luego salida, para que no parpadee al arrancar
  }
  ultimoMensaje = millis();
  Serial.println(F("Listo: esperando numeros por Bluetooth"));
}

void loop() {
  while (bt.available()) {
    char c = bt.read();
    if (c == '\n' || c == '\r') {
      if (largo > 0) {
        linea[largo] = '\0';
        procesarLinea();
      }
    } else if (largo < sizeof(linea) - 1) {
      linea[largo++] = c;
    } else {
      largo = 0;                     // línea demasiado larga: se descarta
    }
  }

  if (mascaraActual != 0 && millis() - ultimoMensaje > TIEMPO_SIN_SENAL_MS) {
    aplicarMascara(0);
    Serial.println(F("Sin senal: focos apagados"));
  }
}
