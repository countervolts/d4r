# d4r для Windows / RDNA4

Windows-форк [countervolts/d4r](https://github.com/countervolts/d4r) для запуска
DLSS Super Resolution на **AMD Radeon RX 9070 XT / gfx1201** через D3D12,
OptiScaler, ZLUDA и HIP. Целевые модели: **DLSS 4 preset K** и
**DLSS 4.5 preset M**. Runtime работает непосредственно в Windows 11.

**Статус: development / prerelease.** K и M работают на тестовой RX 9070 XT,
включая 4K в Silent Hill 2. Текущий приоритет — производительность K.

## Скачать и запустить

[Готовые сборки](https://github.com/xdfnx-dev/d4r/releases) содержат Windows
shim, исправленный ZLUDA, изолированный HIP runtime, patched OptiScaler,
11 native K kernels, 5 native M kernels и исходники соответствующего commit.

Понадобятся Windows 11 x64, RX 9070 XT и D3D12 игра без anti-cheat.
Проверенный AMD driver: `32.0.31041.1004`. Другие GPU и драйверы пока
не прошли такую же проверку.

Предоставьте собственные локальные `_nvngx.dll` и `nvngx_dlss.dll`.
Проверенная пара: NGX `32.0.16.1714`, DLSS `310.9.1.0`. Native manifest
проверяет SHA256 DLSS DLL. NVIDIA binaries в сборку не включены.

Распакуйте ZIP и выполните команду из его каталога, подставив свои пути:

```powershell
.\windows-game.ps1 -GameExe "D:\Games\SILENT HILL 2\SHProto\Binaries\Win64\SHProto-Win64-Shipping.exe" -NgxCore "C:\Users\Administrator\d4r\_nvngx.dll" -DlssDll "C:\Users\Administrator\d4r\nvngx_dlss.dll" -Preset 11 -AsyncInterop
```

В игре выберите upscaler, который перехватывает OptiScaler. `-Preset 11`
выбирает K, `-Preset 13` — M. Начинайте проверку M без `-AsyncInterop`:
асинхронный путь сейчас проверяется прежде всего на K. Первый запуск может
занять время на PTX compilation; следующие используют cache.

Скрипт устанавливает `dxgi.dll`, конфигурацию OptiScaler и каталог `d4r`
рядом с EXE, предварительно сохраняя заменяемые файлы. Вход и выход текущего
кадра остаются в VRAM. Восстановление исходных файлов:

```powershell
.\windows-game.ps1 -Action restore -GameExe "D:\Games\SILENT HILL 2\SHProto\Binaries\Win64\SHProto-Win64-Shipping.exe"
```

Подробные параметры установки и восстановления:
[docs/windows-game.md](docs/windows-game.md).

## Что проверено

| Проверка | Результат на RX 9070 XT |
| --- | --- |
| Обнаружение GPU | HIP автоматически определяет `gfx1201`; проверяется D3D12/HIP LUID |
| HIP и CUDA через ZLUDA | Native HIP kernel и CUDA Driver API / PTX tests проходят |
| D3D12 ↔ HIP | Shared VRAM buffers и fences; import/release lifetime test проходит |
| K transformer | Все 11 native layers, NumPy/PTX/replay validation |
| M transformer | Все 5 native layers, 40 temporal captures, exact baseline |
| Async K | Очередь кадров и Release/CreateFeature точно совпадают с synchronous RGB |
| Silent Hill 2 / K / 4K | 16 941 кадр с проверкой NaN/Inf, 203 292 native launches, 0 backend errors |
| Windows hardware tests | Все 16 CTest gates проходят |

Точные совпадения относятся к проверенным reference/control implementations
этого проекта. Сравнение с DLSS на физической RTX ещё не выполнено.
Проверка одной игры не подтверждает совместимость со всеми D3D12 играми.

## Производительность и ограничения

В локальной 4K сцене пользователь наблюдает около **62 FPS / 63% GPU** с
async interop, против **49–51 FPS / 53%** в предыдущем synchronous варианте.
Замеры также используют локально построенные и проверенные K output-store
kernels. Они содержат NVIDIA-derived code и в публичный ZIP не входят;
команды их локальной сборки есть в
[docs/windows-performance.md](docs/windows-performance.md).

Производительность K ещё дорабатывается. Замеры не являются контролируемым
сравнением с FSR4. `-AsyncInterop` опционален и требует одной D3D12 command
queue. Feature release и resource reconfiguration ожидают завершения GPU
consumers; обычная отправка кадра не удерживает CUDA mutex.

M использует проверенную FP16-equivalent baseline. Native RDNA4 FP8 отдельно
валидирован, но не показал выигрыша для полной сети и выключен по умолчанию.
Windows preset L пока не валидирован. Frame Generation и DLSS 5 не входят
в текущую задачу.

Package использует TheRock `10.2.0a20260929` / HIP `7.17.26386`. Stable HIP
7.2 воспроизводит leak при освобождении mapped external memory; поэтому
закреплён отдельный TheRock runtime. GPU architecture не подменяется.

## Сборка из исходников

```powershell
git clone https://github.com/xdfnx-dev/d4r.git
cd d4r
```

Подготовка закреплённых toolchains и patches описана в
[docs/windows-rdna4-port.md](docs/windows-rdna4-port.md). Нужны CMake + Ninja,
LLVM/MinGW, Rust GNU toolchain, TheRock HIP, а для OptiScaler — MSVC v143
и Windows SDK. Setup-скрипты используют `.tools`; source dependencies — `external`.

После подготовки зависимостей выполните из корня репозитория:

```powershell
.\scripts\windows\build-zluda-windows.ps1
.\scripts\windows\build-windows-rdna4.ps1 -RuntimeProfile therock -ZludaRoot "$PWD\dist\zluda-windows-final" -InstallDirectory "$PWD\dist\windows-rdna4-command-list"
.\scripts\windows\stage-native-k.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -PackageRoot "$PWD\dist\windows-rdna4-command-list"
.\scripts\windows\stage-native-m.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -PackageRoot "$PWD\dist\windows-rdna4-command-list"
.\scripts\windows\build-optiscaler-windows.ps1
.\scripts\windows\package-windows-game.ps1 -ArchivePath "$PWD\dist\windows-rdna4-game.zip"
```

Создание distributable ZIP требует committed source. Package включает manifest
с hashes, dependency licenses, source patches и snapshot этого commit.

## Тесты и диагностика

После сборки hardware gates:

```powershell
& .\.tools\python\cmake\data\bin\ctest.exe --test-dir build/windows-rdna4-therock --output-on-failure
```

Async K regression с локальными NVIDIA DLL в корне репозитория:

```powershell
.\scripts\windows\test-async-k.ps1 -ZludaRoot "$PWD\dist\zluda-windows-final"
```

Успех: `PASS`, exact RGB, finite output и `passed: true` в `validation.json`.
Для ограниченного game test добавьте `-RunSeconds 120 -ValidateOutput` к
команде запуска. Скрипт сохраняет один ZIP с stdout/stderr, runtime/driver
information, native/translated kernel records и OptiScaler logs. Для crash
диагностики используйте `-CaptureExceptions`.

`-ProfileStages` измеряет host stages, `-ProfileCudaApi` — существующие CUDA
API waits, `-ProfileGpuBoundary` вместе с `-AsyncInterop` — D3D12 GPU intervals
вокруг interop. `-ProfileKernels` синхронизирует HIP events и меняет scheduling.
При замере обычного FPS оставляйте profiling и output scans выключенными.

## Документация и лицензии

- [Архитектура, зависимости, milestones и результаты](docs/windows-rdna4-port.md).
- [Game install / restore](docs/windows-game.md).
- [Измерения K/M и FP8](docs/windows-performance.md).
- [OptiScaler source patches](patches/optiscaler/README.md).
- [ZLUDA source patches](patches/zluda).

Проект основан на работе countervolts; Linux upstream code и документация
сохранены для совместимости. Форк не связан с NVIDIA или AMD. Лицензия d4r:
[LICENSE](LICENSE); лицензии зависимостей включены в package.
