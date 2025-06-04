#define CINTERFACE
#include "framework.h"
#include "PupTester.h"
#include <windows.h>
#include <objbase.h>
#include <oleauto.h>
#include <stdio.h>

CLSID pt_clsid = GUID_NULL;
IDispatch* pt_pDispatch = NULL;
DISPID pt_dispidInit = NULL, pt_dispidData = NULL;


const char* pt_Init()
{
    // 2. Get CLSID from ProgID ("PinUpPlayer.PinDisplay")
    if (pt_clsid == GUID_NULL)
    {
        HRESULT hr = CLSIDFromProgID(L"PinUpPlayer.PinDisplay", &pt_clsid);
        if (FAILED(hr))
        {
            pt_clsid = GUID_NULL;
            return "CLSIDFromProgID failed";
        }
    }

    // 3. Create the COM object
    if (pt_pDispatch == NULL)
    {
        HRESULT hr = CoCreateInstance(pt_clsid, NULL, CLSCTX_INPROC_SERVER | CLSCTX_LOCAL_SERVER, IID_IDispatch, (void**)&pt_pDispatch);
        if (FAILED(hr))
        {
            pt_pDispatch = NULL;
            return "CoCreateInstance failed";
        }
    }
    if (pt_dispidInit == NULL)
    {
        OLECHAR* methodName = (wchar_t*)L"B2SInit";
        // Get DISPID from method name
        HRESULT hr = pt_pDispatch->lpVtbl->GetIDsOfNames(pt_pDispatch, IID_NULL, &methodName, 1, LOCALE_USER_DEFAULT, &pt_dispidInit);
        if (FAILED(hr))
        {
            static char tbuf[256];
            sprintf_s(tbuf, sizeof(tbuf), "GetIDsOfNames failed for B2SInit: 0x%08lx", hr);
            return tbuf;
        }
    }
    if (pt_dispidData == NULL)
    {
        OLECHAR* methodName = (wchar_t*)L"B2SData";
        // Get DISPID from method name
        HRESULT hr = pt_pDispatch->lpVtbl->GetIDsOfNames(pt_pDispatch, IID_NULL, &methodName, 1, LOCALE_USER_DEFAULT, &pt_dispidData);
        if (FAILED(hr))
        {
            static char tbuf[256];
            sprintf_s(tbuf, sizeof(tbuf), "GetIDsOfNames failed for B2SInit: 0x%08lx", hr);
            return tbuf;
        }
    }

    return "";
}

const char* pt_CallOleMethod(int function, const VARIANT* pArg1, const VARIANT* pArg2)
{
    if (!pt_pDispatch) return "Invalid IDispatch in pt_CallOleMethod";

    HRESULT hr;
    VARIANTARG args[2];
    DISPPARAMS dp;
    VARIANT result;
    char* error = NULL;

    // Setup arguments (right-to-left order!)
    args[1] = *pArg1; // Copy by value
    args[0] = *pArg2; // Copy by value

    dp.rgvarg = args;
    dp.rgdispidNamedArgs = NULL;
    dp.cArgs = 2;
    dp.cNamedArgs = 0;

    VariantInit(&result);
    DISPID dispid;
    if (function == 0) dispid = pt_dispidInit;
    else if (function == 1) dispid = pt_dispidData;
    else return "Bad function # argument in pt_CallOleMethod";
    if (dispid == NULL) return "Invalid dispid in pt_CallOleMethod";
    // Call the method
    hr = pt_pDispatch->lpVtbl->Invoke(pt_pDispatch, dispid, IID_NULL, LOCALE_USER_DEFAULT,
        DISPATCH_METHOD, &dp, &result, NULL, NULL);
    if (FAILED(hr)) {
        static char tbuf[256];
        sprintf_s(tbuf, sizeof(tbuf), "Invoke failed calling function %i in pt_CallOleMethod", function);
        return tbuf;
    }
    return "";
}

const char* pt_B2sInit(wchar_t* puppackname)
{
    VARIANT arg1, arg2;
    VariantInit(&arg1);
    arg1.vt = VT_BSTR;
    arg1.bstrVal = SysAllocString(L"");  // empty string

    VariantInit(&arg2);
    arg2.vt = VT_BSTR;
    arg2.bstrVal = SysAllocString(puppackname);  // your wide string

    const char* error = pt_CallOleMethod(0, &arg1, &arg2);
    VariantClear(&arg1);
    VariantClear(&arg2);
    return error;
}

const char* pt_B2sData(int IdNum)
{
    wchar_t param1[32];
    swprintf(param1, 32, L"D%d", IdNum); // make the "D123" style string

    VARIANT arg1, arg2;

    VariantInit(&arg1);
    arg1.vt = VT_BSTR;
    arg1.bstrVal = SysAllocString(param1);  // "D123"

    VariantInit(&arg2);
    arg2.vt = VT_I4;
    arg2.lVal = 1; // always 1 in your case

    const char* error = pt_CallOleMethod(1, &arg1, &arg2);
    VariantClear(&arg1);
    VariantClear(&arg2);

    return error;
}

void pt_Close()
{
    if (pt_pDispatch)
    {
        pt_pDispatch->lpVtbl->Release(pt_pDispatch);
        pt_pDispatch = NULL;
    }
}
