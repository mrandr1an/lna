#include <stdio.h>
#include "esp_log.h"

#include "hello.h"
#include "lna_common.h"

static const char *TAG = "lna_basic";

void app_main(void)
{
    ESP_LOGI(TAG, "Starting...");

    printf("%s, ",hello());
    printf("%s\n",name());
    
    ESP_LOGI(TAG, "Done.");
}
