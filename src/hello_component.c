#include "hello_component.h"
#include "esp_log.h"

static const char *TAG = "hello_component";

void hello_component_print(void)
{
    ESP_LOGI(TAG, "Hello from the component repo!");
}
