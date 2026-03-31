#include "xune_audio/xune_nowplaying.h"

#include <dbus/dbus.h>

#include <cstring>
#include <mutex>
#include <thread>
#include <atomic>
#include <string>
#include <cstdio>
#include <unistd.h>

// ── Constants ────────────────────────────────────────────────────────────────

static constexpr const char* MPRIS_BUS_NAME        = "org.mpris.MediaPlayer2.Xune";
static constexpr const char* MPRIS_OBJECT_PATH      = "/org/mpris/MediaPlayer2";
static constexpr const char* MPRIS_IFACE            = "org.mpris.MediaPlayer2";
static constexpr const char* MPRIS_PLAYER_IFACE     = "org.mpris.MediaPlayer2.Player";
static constexpr const char* DBUS_PROPERTIES_IFACE  = "org.freedesktop.DBus.Properties";
static constexpr const char* DBUS_INTROSPECTABLE_IFACE = "org.freedesktop.DBus.Introspectable";

// ── Introspection XML ────────────────────────────────────────────────────────

static constexpr const char* INTROSPECTION_XML =
    "<!DOCTYPE node PUBLIC \"-//freedesktop//DTD D-BUS Object Introspection 1.0//EN\"\n"
    "  \"http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd\">\n"
    "<node>\n"
    "  <interface name=\"org.mpris.MediaPlayer2\">\n"
    "    <method name=\"Raise\"/>\n"
    "    <method name=\"Quit\"/>\n"
    "    <property name=\"CanQuit\" type=\"b\" access=\"read\"/>\n"
    "    <property name=\"CanRaise\" type=\"b\" access=\"read\"/>\n"
    "    <property name=\"HasTrackList\" type=\"b\" access=\"read\"/>\n"
    "    <property name=\"Identity\" type=\"s\" access=\"read\"/>\n"
    "    <property name=\"DesktopEntry\" type=\"s\" access=\"read\"/>\n"
    "    <property name=\"SupportedUriSchemes\" type=\"as\" access=\"read\"/>\n"
    "    <property name=\"SupportedMimeTypes\" type=\"as\" access=\"read\"/>\n"
    "  </interface>\n"
    "  <interface name=\"org.mpris.MediaPlayer2.Player\">\n"
    "    <method name=\"Next\"/>\n"
    "    <method name=\"Previous\"/>\n"
    "    <method name=\"Pause\"/>\n"
    "    <method name=\"PlayPause\"/>\n"
    "    <method name=\"Stop\"/>\n"
    "    <method name=\"Play\"/>\n"
    "    <method name=\"Seek\">\n"
    "      <arg direction=\"in\" name=\"Offset\" type=\"x\"/>\n"
    "    </method>\n"
    "    <method name=\"SetPosition\">\n"
    "      <arg direction=\"in\" name=\"TrackId\" type=\"o\"/>\n"
    "      <arg direction=\"in\" name=\"Position\" type=\"x\"/>\n"
    "    </method>\n"
    "    <property name=\"PlaybackStatus\" type=\"s\" access=\"read\"/>\n"
    "    <property name=\"Rate\" type=\"d\" access=\"readwrite\"/>\n"
    "    <property name=\"Metadata\" type=\"a{sv}\" access=\"read\"/>\n"
    "    <property name=\"Volume\" type=\"d\" access=\"readwrite\"/>\n"
    "    <property name=\"Position\" type=\"x\" access=\"read\"/>\n"
    "    <property name=\"MinimumRate\" type=\"d\" access=\"read\"/>\n"
    "    <property name=\"MaximumRate\" type=\"d\" access=\"read\"/>\n"
    "    <property name=\"CanGoNext\" type=\"b\" access=\"read\"/>\n"
    "    <property name=\"CanGoPrevious\" type=\"b\" access=\"read\"/>\n"
    "    <property name=\"CanPlay\" type=\"b\" access=\"read\"/>\n"
    "    <property name=\"CanPause\" type=\"b\" access=\"read\"/>\n"
    "    <property name=\"CanSeek\" type=\"b\" access=\"read\"/>\n"
    "    <property name=\"CanControl\" type=\"b\" access=\"read\"/>\n"
    "    <signal name=\"Seeked\">\n"
    "      <arg name=\"Position\" type=\"x\"/>\n"
    "    </signal>\n"
    "  </interface>\n"
    "  <interface name=\"org.freedesktop.DBus.Properties\">\n"
    "    <method name=\"Get\">\n"
    "      <arg direction=\"in\" name=\"interface\" type=\"s\"/>\n"
    "      <arg direction=\"in\" name=\"property\" type=\"s\"/>\n"
    "      <arg direction=\"out\" name=\"value\" type=\"v\"/>\n"
    "    </method>\n"
    "    <method name=\"GetAll\">\n"
    "      <arg direction=\"in\" name=\"interface\" type=\"s\"/>\n"
    "      <arg direction=\"out\" name=\"properties\" type=\"a{sv}\"/>\n"
    "    </method>\n"
    "    <signal name=\"PropertiesChanged\">\n"
    "      <arg name=\"interface\" type=\"s\"/>\n"
    "      <arg name=\"changed_properties\" type=\"a{sv}\"/>\n"
    "      <arg name=\"invalidated_properties\" type=\"as\"/>\n"
    "    </signal>\n"
    "  </interface>\n"
    "  <interface name=\"org.freedesktop.DBus.Introspectable\">\n"
    "    <method name=\"Introspect\">\n"
    "      <arg direction=\"out\" name=\"xml\" type=\"s\"/>\n"
    "    </method>\n"
    "  </interface>\n"
    "</node>\n";

// ── Global State ─────────────────────────────────────────────────────────────

static std::recursive_mutex g_mutex;
static DBusConnection* g_conn = nullptr;
static std::thread g_dbus_thread;
static std::atomic<bool> g_running{false};
static std::atomic<bool> g_initialized{false};

static xune_command_callback_t g_callback = nullptr;
static void* g_user_data = nullptr;

static std::string g_playback_status = "Stopped";
static float g_playback_rate = 1.0f;
static int64_t g_position_us = 0;     // microseconds (MPRIS uses µs)
static int64_t g_duration_us = 0;

static std::string g_title;
static std::string g_artist;
static std::string g_album;
static std::string g_album_artist;
static std::string g_art_url;
static uint64_t g_track_id_counter = 0;
static std::string g_track_object_path = "/org/mpris/MediaPlayer2/TrackList/NoTrack";

static bool g_can_play     = true;
static bool g_can_pause    = true;
static bool g_can_next     = true;
static bool g_can_previous = true;
static bool g_can_seek     = true;

// ── Helpers ──────────────────────────────────────────────────────────────────

static void dispatch_command(xune_media_command_t cmd, int64_t param = 0) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_callback) {
        g_callback(cmd, param, g_user_data);
    }
}

/// MPRIS metadata requires a URI for artwork — raw bytes aren't supported.
/// Uses PID in the filename to avoid cross-process clobber.
static std::string write_artwork_to_temp(const void* data, size_t size) {
    if (!data || size == 0) return {};

    auto bytes = static_cast<const uint8_t*>(data);
    const char* ext = (size >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8) ? ".jpg" : ".png";

    std::string path = "/tmp/xune-mpris-artwork-" + std::to_string(getpid()) + ext;

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return {};
    fwrite(data, 1, size, f);
    fclose(f);

    return "file://" + path;
}

// ── D-Bus Message Helpers ────────────────────────────────────────────────────

static void append_variant_string(DBusMessageIter* iter, const char* value) {
    DBusMessageIter variant;
    dbus_message_iter_open_container(iter, DBUS_TYPE_VARIANT, "s", &variant);
    dbus_message_iter_append_basic(&variant, DBUS_TYPE_STRING, &value);
    dbus_message_iter_close_container(iter, &variant);
}

static void append_variant_bool(DBusMessageIter* iter, dbus_bool_t value) {
    DBusMessageIter variant;
    dbus_message_iter_open_container(iter, DBUS_TYPE_VARIANT, "b", &variant);
    dbus_message_iter_append_basic(&variant, DBUS_TYPE_BOOLEAN, &value);
    dbus_message_iter_close_container(iter, &variant);
}

static void append_variant_int64(DBusMessageIter* iter, int64_t value) {
    dbus_int64_t v = static_cast<dbus_int64_t>(value);
    DBusMessageIter variant;
    dbus_message_iter_open_container(iter, DBUS_TYPE_VARIANT, "x", &variant);
    dbus_message_iter_append_basic(&variant, DBUS_TYPE_INT64, &v);
    dbus_message_iter_close_container(iter, &variant);
}

static void append_variant_double(DBusMessageIter* iter, double value) {
    DBusMessageIter variant;
    dbus_message_iter_open_container(iter, DBUS_TYPE_VARIANT, "d", &variant);
    dbus_message_iter_append_basic(&variant, DBUS_TYPE_DOUBLE, &value);
    dbus_message_iter_close_container(iter, &variant);
}

static void append_variant_empty_string_array(DBusMessageIter* iter) {
    DBusMessageIter variant, array;
    dbus_message_iter_open_container(iter, DBUS_TYPE_VARIANT, "as", &variant);
    dbus_message_iter_open_container(&variant, DBUS_TYPE_ARRAY, "s", &array);
    dbus_message_iter_close_container(&variant, &array);
    dbus_message_iter_close_container(iter, &variant);
}

static void append_variant_string_array(DBusMessageIter* iter, const char* value) {
    DBusMessageIter variant, array;
    dbus_message_iter_open_container(iter, DBUS_TYPE_VARIANT, "as", &variant);
    dbus_message_iter_open_container(&variant, DBUS_TYPE_ARRAY, "s", &array);
    dbus_message_iter_append_basic(&array, DBUS_TYPE_STRING, &value);
    dbus_message_iter_close_container(&variant, &array);
    dbus_message_iter_close_container(iter, &variant);
}

// ── Dict Entry Helpers (a{sv} building blocks) ──────────────────────────────

static void append_dict_sv_string(DBusMessageIter* array, const char* key, const char* value) {
    DBusMessageIter entry;
    dbus_message_iter_open_container(array, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    append_variant_string(&entry, value);
    dbus_message_iter_close_container(array, &entry);
}

static void append_dict_sv_bool(DBusMessageIter* array, const char* key, bool value) {
    DBusMessageIter entry;
    dbus_message_iter_open_container(array, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    dbus_bool_t b = value ? TRUE : FALSE;
    append_variant_bool(&entry, b);
    dbus_message_iter_close_container(array, &entry);
}

static void append_dict_sv_int64(DBusMessageIter* array, const char* key, int64_t value) {
    DBusMessageIter entry;
    dbus_message_iter_open_container(array, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    append_variant_int64(&entry, value);
    dbus_message_iter_close_container(array, &entry);
}

static void append_dict_sv_double(DBusMessageIter* array, const char* key, double value) {
    DBusMessageIter entry;
    dbus_message_iter_open_container(array, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    append_variant_double(&entry, value);
    dbus_message_iter_close_container(array, &entry);
}

static void append_dict_sv_string_array(DBusMessageIter* array, const char* key, const char* value) {
    DBusMessageIter entry;
    dbus_message_iter_open_container(array, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    append_variant_string_array(&entry, value);
    dbus_message_iter_close_container(array, &entry);
}

static void append_dict_sv_empty_string_array(DBusMessageIter* array, const char* key) {
    DBusMessageIter entry;
    dbus_message_iter_open_container(array, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    append_variant_empty_string_array(&entry);
    dbus_message_iter_close_container(array, &entry);
}

static void append_dict_sv_object_path(DBusMessageIter* array_iter, const char* key, const char* path) {
    DBusMessageIter entry, variant;
    dbus_message_iter_open_container(array_iter, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    dbus_message_iter_open_container(&entry, DBUS_TYPE_VARIANT, "o", &variant);
    dbus_message_iter_append_basic(&variant, DBUS_TYPE_OBJECT_PATH, &path);
    dbus_message_iter_close_container(&entry, &variant);
    dbus_message_iter_close_container(array_iter, &entry);
}

// ── Metadata ─────────────────────────────────────────────────────────────────

/// Build the Metadata dict (a{sv}) into an open variant iterator
static void append_metadata_variant(DBusMessageIter* iter) {
    DBusMessageIter variant, array;
    dbus_message_iter_open_container(iter, DBUS_TYPE_VARIANT, "a{sv}", &variant);
    dbus_message_iter_open_container(&variant, DBUS_TYPE_ARRAY, "{sv}", &array);

    // mpris:trackid is required
    append_dict_sv_object_path(&array, "mpris:trackid", g_track_object_path.c_str());

    if (g_duration_us > 0)
        append_dict_sv_int64(&array, "mpris:length", g_duration_us);
    if (!g_title.empty())
        append_dict_sv_string(&array, "xesam:title", g_title.c_str());
    if (!g_artist.empty())
        append_dict_sv_string_array(&array, "xesam:artist", g_artist.c_str());
    if (!g_album.empty())
        append_dict_sv_string(&array, "xesam:album", g_album.c_str());
    if (!g_album_artist.empty())
        append_dict_sv_string_array(&array, "xesam:albumArtist", g_album_artist.c_str());
    if (!g_art_url.empty())
        append_dict_sv_string(&array, "mpris:artUrl", g_art_url.c_str());

    dbus_message_iter_close_container(&variant, &array);
    dbus_message_iter_close_container(iter, &variant);
}

// ── PropertiesChanged Signal ─────────────────────────────────────────────────

/// Signal envelope for PropertiesChanged. Opens the changed-properties dict;
/// caller populates it with append_dict_sv_* helpers, then calls _finish.
struct PropsChangedSignal {
    DBusMessage* sig;
    DBusMessageIter msg_iter;
    DBusMessageIter dict_iter;
};

static bool props_changed_begin(PropsChangedSignal& s) {
    if (!g_conn) return false;

    s.sig = dbus_message_new_signal(
        MPRIS_OBJECT_PATH, DBUS_PROPERTIES_IFACE, "PropertiesChanged");
    if (!s.sig) return false;

    dbus_message_iter_init_append(s.sig, &s.msg_iter);
    const char* iface = MPRIS_PLAYER_IFACE;
    dbus_message_iter_append_basic(&s.msg_iter, DBUS_TYPE_STRING, &iface);
    dbus_message_iter_open_container(&s.msg_iter, DBUS_TYPE_ARRAY, "{sv}", &s.dict_iter);
    return true;
}

static void props_changed_finish(PropsChangedSignal& s) {
    dbus_message_iter_close_container(&s.msg_iter, &s.dict_iter);

    DBusMessageIter invalidated;
    dbus_message_iter_open_container(&s.msg_iter, DBUS_TYPE_ARRAY, "s", &invalidated);
    dbus_message_iter_close_container(&s.msg_iter, &invalidated);

    dbus_connection_send(g_conn, s.sig, nullptr);
    dbus_message_unref(s.sig);
}

// ── Property Getters ─────────────────────────────────────────────────────────

static DBusMessage* handle_get_root_property(DBusMessage* msg, const char* property) {
    DBusMessage* reply = dbus_message_new_method_return(msg);
    DBusMessageIter iter;
    dbus_message_iter_init_append(reply, &iter);

    if (strcmp(property, "CanQuit") == 0) {
        dbus_bool_t val = FALSE;
        append_variant_bool(&iter, val);
    } else if (strcmp(property, "CanRaise") == 0) {
        dbus_bool_t val = FALSE;
        append_variant_bool(&iter, val);
    } else if (strcmp(property, "HasTrackList") == 0) {
        dbus_bool_t val = FALSE;
        append_variant_bool(&iter, val);
    } else if (strcmp(property, "Identity") == 0) {
        append_variant_string(&iter, "Xune");
    } else if (strcmp(property, "DesktopEntry") == 0) {
        append_variant_string(&iter, "xune");
    } else if (strcmp(property, "SupportedUriSchemes") == 0) {
        append_variant_empty_string_array(&iter);
    } else if (strcmp(property, "SupportedMimeTypes") == 0) {
        append_variant_empty_string_array(&iter);
    } else {
        dbus_message_unref(reply);
        return dbus_message_new_error(msg, DBUS_ERROR_UNKNOWN_PROPERTY, property);
    }

    return reply;
}

static DBusMessage* handle_get_player_property(DBusMessage* msg, const char* property) {
    DBusMessage* reply = dbus_message_new_method_return(msg);
    DBusMessageIter iter;
    dbus_message_iter_init_append(reply, &iter);

    if (strcmp(property, "PlaybackStatus") == 0) {
        append_variant_string(&iter, g_playback_status.c_str());
    } else if (strcmp(property, "Rate") == 0) {
        append_variant_double(&iter, static_cast<double>(g_playback_rate));
    } else if (strcmp(property, "Metadata") == 0) {
        append_metadata_variant(&iter);
    } else if (strcmp(property, "Volume") == 0) {
        append_variant_double(&iter, 1.0);
    } else if (strcmp(property, "Position") == 0) {
        append_variant_int64(&iter, g_position_us);
    } else if (strcmp(property, "MinimumRate") == 0) {
        append_variant_double(&iter, 1.0);
    } else if (strcmp(property, "MaximumRate") == 0) {
        append_variant_double(&iter, 1.0);
    } else if (strcmp(property, "CanGoNext") == 0) {
        dbus_bool_t val = g_can_next ? TRUE : FALSE;
        append_variant_bool(&iter, val);
    } else if (strcmp(property, "CanGoPrevious") == 0) {
        dbus_bool_t val = g_can_previous ? TRUE : FALSE;
        append_variant_bool(&iter, val);
    } else if (strcmp(property, "CanPlay") == 0) {
        dbus_bool_t val = g_can_play ? TRUE : FALSE;
        append_variant_bool(&iter, val);
    } else if (strcmp(property, "CanPause") == 0) {
        dbus_bool_t val = g_can_pause ? TRUE : FALSE;
        append_variant_bool(&iter, val);
    } else if (strcmp(property, "CanSeek") == 0) {
        dbus_bool_t val = g_can_seek ? TRUE : FALSE;
        append_variant_bool(&iter, val);
    } else if (strcmp(property, "CanControl") == 0) {
        dbus_bool_t val = TRUE;
        append_variant_bool(&iter, val);
    } else {
        dbus_message_unref(reply);
        return dbus_message_new_error(msg, DBUS_ERROR_UNKNOWN_PROPERTY, property);
    }

    return reply;
}

static void append_all_player_properties(DBusMessageIter* array) {
    append_dict_sv_string(array, "PlaybackStatus", g_playback_status.c_str());
    append_dict_sv_double(array, "Rate", static_cast<double>(g_playback_rate));

    DBusMessageIter entry;
    dbus_message_iter_open_container(array, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
    const char* key = "Metadata";
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    append_metadata_variant(&entry);
    dbus_message_iter_close_container(array, &entry);

    append_dict_sv_double(array, "Volume", 1.0);
    append_dict_sv_int64(array, "Position", g_position_us);
    append_dict_sv_bool(array, "CanGoNext", g_can_next);
    append_dict_sv_bool(array, "CanGoPrevious", g_can_previous);
    append_dict_sv_bool(array, "CanPlay", g_can_play);
    append_dict_sv_bool(array, "CanPause", g_can_pause);
    append_dict_sv_bool(array, "CanSeek", g_can_seek);
    append_dict_sv_bool(array, "CanControl", true);
    append_dict_sv_double(array, "MinimumRate", 1.0);
    append_dict_sv_double(array, "MaximumRate", 1.0);
}

static void append_all_root_properties(DBusMessageIter* array) {
    append_dict_sv_bool(array, "CanQuit", false);
    append_dict_sv_bool(array, "CanRaise", false);
    append_dict_sv_bool(array, "HasTrackList", false);
    append_dict_sv_string(array, "Identity", "Xune");
    append_dict_sv_string(array, "DesktopEntry", "xune");
    append_dict_sv_empty_string_array(array, "SupportedUriSchemes");
    append_dict_sv_empty_string_array(array, "SupportedMimeTypes");
}

// ── D-Bus Message Handler ────────────────────────────────────────────────────

static DBusHandlerResult handle_message(
    DBusConnection* conn, DBusMessage* msg, void* /*user_data*/)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    const char* iface = dbus_message_get_interface(msg);
    const char* member = dbus_message_get_member(msg);

    if (!iface || !member) return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;

    if (dbus_message_is_method_call(msg, DBUS_INTROSPECTABLE_IFACE, "Introspect")) {
        DBusMessage* reply = dbus_message_new_method_return(msg);
        dbus_message_append_args(reply,
            DBUS_TYPE_STRING, &INTROSPECTION_XML,
            DBUS_TYPE_INVALID);
        dbus_connection_send(conn, reply, nullptr);
        dbus_message_unref(reply);
        return DBUS_HANDLER_RESULT_HANDLED;
    }

    if (strcmp(iface, DBUS_PROPERTIES_IFACE) == 0) {
        if (strcmp(member, "Get") == 0) {
            const char* prop_iface = nullptr;
            const char* prop_name = nullptr;
            if (!dbus_message_get_args(msg, nullptr,
                    DBUS_TYPE_STRING, &prop_iface,
                    DBUS_TYPE_STRING, &prop_name,
                    DBUS_TYPE_INVALID)) {
                return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
            }

            DBusMessage* reply = nullptr;
            if (strcmp(prop_iface, MPRIS_IFACE) == 0) {
                reply = handle_get_root_property(msg, prop_name);
            } else if (strcmp(prop_iface, MPRIS_PLAYER_IFACE) == 0) {
                reply = handle_get_player_property(msg, prop_name);
            } else {
                reply = dbus_message_new_error(msg,
                    DBUS_ERROR_UNKNOWN_INTERFACE, prop_iface);
            }
            dbus_connection_send(conn, reply, nullptr);
            dbus_message_unref(reply);
            return DBUS_HANDLER_RESULT_HANDLED;
        }

        if (strcmp(member, "GetAll") == 0) {
            const char* prop_iface = nullptr;
            if (!dbus_message_get_args(msg, nullptr,
                    DBUS_TYPE_STRING, &prop_iface,
                    DBUS_TYPE_INVALID)) {
                return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
            }

            DBusMessage* reply = dbus_message_new_method_return(msg);
            DBusMessageIter iter, array;
            dbus_message_iter_init_append(reply, &iter);
            dbus_message_iter_open_container(&iter, DBUS_TYPE_ARRAY, "{sv}", &array);

            if (strcmp(prop_iface, MPRIS_PLAYER_IFACE) == 0) {
                append_all_player_properties(&array);
            } else if (strcmp(prop_iface, MPRIS_IFACE) == 0) {
                append_all_root_properties(&array);
            }

            dbus_message_iter_close_container(&iter, &array);
            dbus_connection_send(conn, reply, nullptr);
            dbus_message_unref(reply);
            return DBUS_HANDLER_RESULT_HANDLED;
        }

        if (strcmp(member, "Set") == 0) {
            DBusMessage* reply = dbus_message_new_method_return(msg);
            dbus_connection_send(conn, reply, nullptr);
            dbus_message_unref(reply);
            return DBUS_HANDLER_RESULT_HANDLED;
        }
    }

    if (strcmp(iface, MPRIS_IFACE) == 0) {
        if (strcmp(member, "Raise") == 0 || strcmp(member, "Quit") == 0) {
            DBusMessage* reply = dbus_message_new_method_return(msg);
            dbus_connection_send(conn, reply, nullptr);
            dbus_message_unref(reply);
            return DBUS_HANDLER_RESULT_HANDLED;
        }
    }

    if (strcmp(iface, MPRIS_PLAYER_IFACE) == 0) {
        DBusMessage* reply = dbus_message_new_method_return(msg);

        if (strcmp(member, "Play") == 0) {
            dispatch_command(XUNE_CMD_PLAY);
        } else if (strcmp(member, "Pause") == 0) {
            dispatch_command(XUNE_CMD_PAUSE);
        } else if (strcmp(member, "PlayPause") == 0) {
            dispatch_command(XUNE_CMD_TOGGLE_PLAY_PAUSE);
        } else if (strcmp(member, "Stop") == 0) {
            dispatch_command(XUNE_CMD_STOP);
        } else if (strcmp(member, "Next") == 0) {
            dispatch_command(XUNE_CMD_NEXT);
        } else if (strcmp(member, "Previous") == 0) {
            dispatch_command(XUNE_CMD_PREVIOUS);
        } else if (strcmp(member, "Seek") == 0) {
            // MPRIS Seek is a relative offset in µs — clamp and convert to absolute ms
            dbus_int64_t offset_us = 0;
            if (dbus_message_get_args(msg, nullptr,
                    DBUS_TYPE_INT64, &offset_us,
                    DBUS_TYPE_INVALID)) {
                int64_t new_pos_us = g_position_us + offset_us;
                if (new_pos_us < 0) new_pos_us = 0;
                if (g_duration_us > 0 && new_pos_us > g_duration_us) new_pos_us = g_duration_us;
                dispatch_command(XUNE_CMD_SEEK, new_pos_us / 1000);
            }
        } else if (strcmp(member, "SetPosition") == 0) {
            const char* track_id = nullptr;
            dbus_int64_t position_us = 0;
            if (dbus_message_get_args(msg, nullptr,
                    DBUS_TYPE_OBJECT_PATH, &track_id,
                    DBUS_TYPE_INT64, &position_us,
                    DBUS_TYPE_INVALID)) {
                // MPRIS spec: ignore if track ID doesn't match current track
                if (g_track_object_path == track_id) {
                    dispatch_command(XUNE_CMD_SEEK, position_us / 1000);
                }
            }
        } else {
            dbus_message_unref(reply);
            return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
        }

        dbus_connection_send(conn, reply, nullptr);
        dbus_message_unref(reply);
        return DBUS_HANDLER_RESULT_HANDLED;
    }

    return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}

// ── D-Bus Event Loop Thread ──────────────────────────────────────────────────

static void dbus_loop() {
    while (g_running.load()) {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        if (g_conn) {
            // read_write does socket IO (blocks up to 50ms), dispatch processes
            // queued messages. Both under the mutex to prevent racing cleanup.
            dbus_connection_read_write(g_conn, 50);
            while (dbus_connection_dispatch(g_conn) == DBUS_DISPATCH_DATA_REMAINS) {}
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

int xune_nowplaying_init(
    void* window_handle,
    xune_command_callback_t callback,
    void* user_data)
{
    (void)window_handle;

    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (g_initialized.load()) return 0;

    DBusError err;
    dbus_error_init(&err);

    g_conn = dbus_bus_get(DBUS_BUS_SESSION, &err);
    if (!g_conn || dbus_error_is_set(&err)) {
        if (dbus_error_is_set(&err)) {
            dbus_error_free(&err);
        }
        return -1;
    }

    int ret = dbus_bus_request_name(g_conn, MPRIS_BUS_NAME,
        DBUS_NAME_FLAG_DO_NOT_QUEUE, &err);
    if (ret != DBUS_REQUEST_NAME_REPLY_PRIMARY_OWNER) {
        if (dbus_error_is_set(&err)) {
            dbus_error_free(&err);
        }
        dbus_connection_unref(g_conn);
        g_conn = nullptr;
        return -1;
    }

    static const DBusObjectPathVTable vtable = {
        nullptr,         // unregister function
        handle_message,  // message function
        nullptr, nullptr, nullptr, nullptr
    };

    if (!dbus_connection_register_object_path(g_conn, MPRIS_OBJECT_PATH, &vtable, nullptr)) {
        dbus_bus_release_name(g_conn, MPRIS_BUS_NAME, nullptr);
        dbus_connection_unref(g_conn);
        g_conn = nullptr;
        return -1;
    }

    g_callback = callback;
    g_user_data = user_data;
    g_initialized.store(true);

    g_running.store(true);
    g_dbus_thread = std::thread(dbus_loop);

    return 0;
}

bool xune_nowplaying_is_available(void) {
    DBusError err;
    dbus_error_init(&err);
    DBusConnection* conn = dbus_bus_get(DBUS_BUS_SESSION, &err);
    bool available = (conn != nullptr && !dbus_error_is_set(&err));
    if (dbus_error_is_set(&err)) {
        dbus_error_free(&err);
    }
    // dbus_bus_get returns a shared connection — unref to balance the refcount
    if (conn) {
        dbus_connection_unref(conn);
    }
    return available;
}

void xune_nowplaying_cleanup(void) {
    g_running.store(false);
    if (g_dbus_thread.joinable()) {
        g_dbus_thread.join();
    }

    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized.load()) return;

    g_callback = nullptr;
    g_user_data = nullptr;

    if (g_conn) {
        dbus_connection_unregister_object_path(g_conn, MPRIS_OBJECT_PATH);
        dbus_bus_release_name(g_conn, MPRIS_BUS_NAME, nullptr);
        dbus_connection_unref(g_conn);
        g_conn = nullptr;
    }

    g_playback_status = "Stopped";
    g_position_us = 0;
    g_duration_us = 0;
    g_title.clear();
    g_artist.clear();
    g_album.clear();
    g_album_artist.clear();
    g_art_url.clear();
    g_track_object_path = "/org/mpris/MediaPlayer2/TrackList/NoTrack";
    g_track_id_counter = 0;

    g_initialized.store(false);
}

void xune_nowplaying_set_metadata(const xune_track_metadata_t* metadata) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load() || !metadata) return;

    g_title = metadata->title ? metadata->title : "";
    g_artist = metadata->artist ? metadata->artist : "";
    g_album = metadata->album ? metadata->album : "";
    g_album_artist = metadata->album_artist ? metadata->album_artist : "";
    g_duration_us = metadata->duration_ms * 1000;

    g_track_id_counter++;
    g_track_object_path = "/org/mpris/MediaPlayer2/Track/" + std::to_string(g_track_id_counter);

    g_art_url.clear();
    if (metadata->artwork_data && metadata->artwork_size > 0) {
        g_art_url = write_artwork_to_temp(metadata->artwork_data, metadata->artwork_size);
    } else if (metadata->artwork_path) {
        g_art_url = "file://";
        g_art_url += metadata->artwork_path;
    }

    PropsChangedSignal s;
    if (props_changed_begin(s)) {
        DBusMessageIter entry;
        dbus_message_iter_open_container(&s.dict_iter, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
        const char* key = "Metadata";
        dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
        append_metadata_variant(&entry);
        dbus_message_iter_close_container(&s.dict_iter, &entry);
        props_changed_finish(s);
    }
}

void xune_nowplaying_clear_metadata(void) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load()) return;

    g_title.clear();
    g_artist.clear();
    g_album.clear();
    g_album_artist.clear();
    g_art_url.clear();
    g_duration_us = 0;
    g_position_us = 0;
    g_track_object_path = "/org/mpris/MediaPlayer2/TrackList/NoTrack";
    g_playback_status = "Stopped";

    PropsChangedSignal s;
    if (props_changed_begin(s)) {
        DBusMessageIter entry;
        dbus_message_iter_open_container(&s.dict_iter, DBUS_TYPE_DICT_ENTRY, nullptr, &entry);
        const char* key = "Metadata";
        dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
        append_metadata_variant(&entry);
        dbus_message_iter_close_container(&s.dict_iter, &entry);
        append_dict_sv_string(&s.dict_iter, "PlaybackStatus", g_playback_status.c_str());
        props_changed_finish(s);
    }
}

void xune_nowplaying_set_playback_state(xune_playback_state_t state) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load()) return;

    switch (state) {
        case XUNE_PLAYBACK_PLAYING: g_playback_status = "Playing"; break;
        case XUNE_PLAYBACK_PAUSED:  g_playback_status = "Paused"; break;
        default:                    g_playback_status = "Stopped"; break;
    }

    PropsChangedSignal s;
    if (props_changed_begin(s)) {
        append_dict_sv_string(&s.dict_iter, "PlaybackStatus", g_playback_status.c_str());
        props_changed_finish(s);
    }
}

void xune_nowplaying_set_position(int64_t position_ms, int64_t duration_ms) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load()) return;

    g_position_us = position_ms * 1000;
    g_duration_us = duration_ms * 1000;

    // MPRIS Position is read via Properties.Get, not signaled during normal playback.
    // The Seeked signal is only for discontinuous jumps (user-initiated seek).
}

void xune_nowplaying_set_playback_rate(float rate) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load()) return;

    g_playback_rate = rate;

    PropsChangedSignal s;
    if (props_changed_begin(s)) {
        append_dict_sv_double(&s.dict_iter, "Rate", static_cast<double>(g_playback_rate));
        props_changed_finish(s);
    }
}

void xune_nowplaying_set_command_enabled(xune_media_command_t command, bool enabled) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_initialized.load()) return;

    const char* prop_name = nullptr;
    bool* target = nullptr;

    switch (command) {
        case XUNE_CMD_PLAY:
            prop_name = "CanPlay"; target = &g_can_play; break;
        case XUNE_CMD_PAUSE:
            prop_name = "CanPause"; target = &g_can_pause; break;
        case XUNE_CMD_NEXT:
            prop_name = "CanGoNext"; target = &g_can_next; break;
        case XUNE_CMD_PREVIOUS:
            prop_name = "CanGoPrevious"; target = &g_can_previous; break;
        case XUNE_CMD_SEEK:
            prop_name = "CanSeek"; target = &g_can_seek; break;
        default:
            return;
    }

    if (*target == enabled) return;
    *target = enabled;

    PropsChangedSignal s;
    if (props_changed_begin(s)) {
        append_dict_sv_bool(&s.dict_iter, prop_name, enabled);
        props_changed_finish(s);
    }
}
