// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public enum MlxDeviceType
{
    MlxCpu,
    MlxGpu,
}

public static partial class MlxDevice
{
    /// <summary>
    /// Creates a new device object for the default device type.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_new")]
    public static partial MlxDeviceHandle New();

    /// <summary>
    /// Creates a new device object for a specified device type and index (e.g. a particular GPU).
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_new_type")]
    public static partial MlxDeviceHandle NewType(MlxDeviceType type, int index);

    /// <summary>
    /// Releases a device handle and its associated resources.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_free")]
    public static partial int Free(MlxDeviceHandle dev);

    /// <summary>
    /// Sets the current default device to the given device.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_set")]
    public static partial int Set(ref MlxDeviceHandle dev, MlxDeviceHandle src);

    /// <summary>
    /// Returns a string representation of the device (including its type and index).
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_tostring")]
    public static partial int ToString(out MlxStringHandle str, MlxDeviceHandle dev);

    /// <summary>
    /// Checks if two device handles represent the same device.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_equal")]
    [return: MarshalAs(UnmanagedType.I1)]
    public static partial bool Equal(MlxDeviceHandle lhs, MlxDeviceHandle rhs);

    /// <summary>
    /// Returns the device index of the given device handle.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_get_index")]
    public static partial int GetIndex(out int index, MlxDeviceHandle dev);

    /// <summary>
    /// Returns the type of the device for the given device handle.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_get_type")]
    public static partial int GetType(out MlxDeviceType type, MlxDeviceHandle dev);

    /// <summary>
    /// Returns the current default device on which new arrays and operations are placed.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_get_default_device")]
    public static partial int GetDefaultDevice(out MlxDeviceHandle dev);

    /// <summary>
    /// Sets the default device to the specified device.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_set_default_device")]
    public static partial int SetDefaultDevice(MlxDeviceHandle dev);

    /// <summary>
    /// Indicates whether the specified device is available.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_is_available")]
    public static partial int IsAvailable([MarshalAs(UnmanagedType.I1)] out bool available, MlxDeviceHandle dev);

    /// <summary>
    /// Returns the number of available devices for the specified backend type.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_count")]
    public static partial int Count(out int count, MlxDeviceType type);

    /// <summary>
    /// Creates an empty device info container.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_info_new")]
    public static partial MlxDeviceInfo DeviceInfoNew();

    /// <summary>
    /// Populates device information for the specified device.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_info_get")]
    public static partial int DeviceInfoGet(out MlxDeviceInfo info, MlxDeviceHandle dev);

    /// <summary>
    /// Releases device information resources.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_info_free")]
    public static partial int DeviceInfoFree(MlxDeviceInfo info);

    /// <summary>
    /// Indicates whether a device info key exists.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_info_has_key", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int DeviceInfoHasKey(
        [MarshalAs(UnmanagedType.I1)] out bool exists,
        MlxDeviceInfo info,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>
    /// Indicates whether a device info value is a string.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_info_is_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int DeviceInfoIsString(
        [MarshalAs(UnmanagedType.I1)] out bool isString,
        MlxDeviceInfo info,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>
    /// Returns a UTF-8 string pointer for the specified device info key.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_info_get_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int DeviceInfoGetString(out nint value, MlxDeviceInfo info, [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>
    /// Returns a numeric device info value for the specified key.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_info_get_size", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int DeviceInfoGetSize(out nuint value, MlxDeviceInfo info, [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>
    /// Returns all available device info keys.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_device_info_get_keys")]
    public static partial int DeviceInfoGetKeys(out MlxVectorStringHandle keys, MlxDeviceInfo info);
}
