# Input data

The analysis expects OSM-derived bus-network and POI JSON files. Raw extracts
are not stored in Git because their combined size exceeds GitHub's limits.

By default, place the following files in this directory:

```text
BeiJing_bus.json
BeiJing_pois.json
ShangHai_bus.json
ShangHai_pois.json
HongKong_bus.json
HongKong_pois.json
Москва_bus.json
Москва_pois.json
Санкт-Петербург_bus.json
Санкт-Петербург_pois.json
городской округ Казань_bus.json
городской округ Казань_pois.json
городской округ Екатеринбург_bus.json
городской округ Екатеринбург_pois.json
```

The expected filenames are configured in `config/cities.json`. Relative paths
are resolved under `PTN_DATA_ROOT`, which defaults to this directory.

The files contain OpenStreetMap-derived information and must retain
OpenStreetMap attribution. Users redistributing database extracts should review
the Open Database License and the OSM attribution guidelines.

