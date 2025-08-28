// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <dispatch/dispatch.h>
#import <dispatch/data.h>
#import <mach-o/getsect.h>
#import <mach-o/loader.h>
#import <stdatomic.h>

/*dispatch_data_t data = dispatch_data_create(p, sz, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{});*/

extern const struct mach_header_64 _mh_dylib_header;

static id<MTLLibrary> g_lib;
static dispatch_once_t g_once;
static _Atomic int g_swizzled;

static id<MTLLibrary> make_embedded_lib(id<MTLDevice> dev) {
    dispatch_once(&g_once, ^{
        unsigned long sz = 0;
        const void* p = getsectiondata(&_mh_dylib_header, "__DATA", "__mlx_metallib", &sz);
        if (p && sz > 0) {
            dispatch_data_t data = dispatch_data_create(p, (size_t)sz, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{});
            NSError* err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithData:data error:&err];
            if (lib) g_lib = lib;
            else fprintf(stderr, "MLX metallib load error: %s\n", err.localizedDescription.UTF8String ?: "nil");
        }
    });
    return g_lib;
}

static IMP orig_newDefaultLibrary;
static id hook_newDefaultLibrary(id self, SEL _cmd) {
    id<MTLLibrary> lib = make_embedded_lib((id<MTLDevice>)self);
    if (lib) return lib;
    return ((id(*)(id,SEL))orig_newDefaultLibrary)(self, _cmd);
}

static IMP orig_newDefaultLibraryWithBundle;
static id hook_newDefaultLibraryWithBundle(id self, SEL _cmd, NSBundle* bundle, NSError** error) {
    id<MTLLibrary> lib = make_embedded_lib((id<MTLDevice>)self);
    if (lib) return lib;
    return ((id(*)(id,SEL,NSBundle*,NSError**))orig_newDefaultLibraryWithBundle)(self, _cmd, bundle, error);
}

static IMP orig_newLibraryWithFile;
static id hook_newLibraryWithFile(id self, SEL _cmd, NSString* path, NSError** error) {
    id<MTLLibrary> lib = make_embedded_lib((id<MTLDevice>)self);
    if (lib) return lib;
    return ((id(*)(id,SEL,NSString*,NSError**))orig_newLibraryWithFile)(self, _cmd, path, error);
}

static IMP orig_newLibraryWithURL;
static id hook_newLibraryWithURL(id self, SEL _cmd, NSURL* url, NSError** error) {
    id<MTLLibrary> lib = make_embedded_lib((id<MTLDevice>)self);
    if (lib) return lib;
    return ((id(*)(id,SEL,NSURL*,NSError**))orig_newLibraryWithURL)(self, _cmd, url, error);
}

__attribute__((constructor))
static void mlxc_install_swizzle(void) {
    @autoreleasepool {
        if (atomic_exchange(&g_swizzled, 1)) return;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return;

        Class cls = object_getClass((id)dev);

        SEL s1 = sel_registerName("newDefaultLibrary");
        Method m1 = class_getInstanceMethod(cls, s1);
        if (m1) {
            orig_newDefaultLibrary = method_getImplementation(m1);
            if (orig_newDefaultLibrary != (IMP)hook_newDefaultLibrary)
                method_setImplementation(m1, (IMP)hook_newDefaultLibrary);
        }

        SEL s2 = sel_registerName("newDefaultLibraryWithBundle:error:");
        Method m2 = class_getInstanceMethod(cls, s2);
        if (m2) {
            orig_newDefaultLibraryWithBundle = method_getImplementation(m2);
            if (orig_newDefaultLibraryWithBundle != (IMP)hook_newDefaultLibraryWithBundle)
                method_setImplementation(m2, (IMP)hook_newDefaultLibraryWithBundle);
        }

        SEL s3 = sel_registerName("newLibraryWithFile:error:");
        Method m3 = class_getInstanceMethod(cls, s3);
        if (m3) {
            orig_newLibraryWithFile = method_getImplementation(m3);
            if (orig_newLibraryWithFile != (IMP)hook_newLibraryWithFile)
                method_setImplementation(m3, (IMP)hook_newLibraryWithFile);
        }

        SEL s4 = sel_registerName("newLibraryWithURL:error:");
        Method m4 = class_getInstanceMethod(cls, s4);
        if (m4) {
            orig_newLibraryWithURL = method_getImplementation(m4);
            if (orig_newLibraryWithURL != (IMP)hook_newLibraryWithURL)
                method_setImplementation(m4, (IMP)hook_newLibraryWithURL);
        }
    }
}