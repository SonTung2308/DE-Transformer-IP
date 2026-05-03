from django.contrib import admin
from myapp.models import *
from django.utils.html import format_html
from django import forms
# Register your models here.

  
class adminproduct(admin.ModelAdmin):
    list_display=['name','price','categories']

class admincatagory(admin.ModelAdmin):
    list_display=['name','id']
    
class adminrecipie(admin.ModelAdmin):
    list_display=['name']

class adminorder(admin.ModelAdmin):
    list_display=['orderid','customer','date','price']

class admincon(admin.ModelAdmin):
    list_display=['name','email','desc']

class adminfback(admin.ModelAdmin):
    list_display=['des','score']

class admintotalorder(admin.ModelAdmin):
    list_display=['orderid','name','address','phone','date','totalamount']

class adminprofile(admin.ModelAdmin):
    list_display=['username','email','address','phone']

class adminelections(admin.ModelAdmin):
    list_display=['name','id']

class adminvote(admin.ModelAdmin):
    list_display=['election','candidate','user']


class adminCandidate(admin.ModelAdmin):
    list_display = ['get_election_name', 'candidatename', 'display_photo','vote']
   

    def get_election_name(self, obj):
        return obj.elections.name if obj.elections else 'No Election'
    get_election_name.short_description = 'Election'

    def display_photo(self, obj):
        if obj.image:
            return format_html('<img src="{}" style="width: 100px; height: auto;" />'.format(obj.image.url))
        else:
            return 'No Image'

    display_photo.short_description = 'Photo'

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == 'elections':
            kwargs['queryset'] = elections.objects.all().order_by('name')
        return super().formfield_for_foreignkey(db_field, request, **kwargs)



admin.site.site_header="admin site"

admin.site.register(con,admincon)
admin.site.register(fback,adminfback)
admin.site.register(product,adminproduct)
admin.site.register(catagory,admincatagory)
admin.site.register(Order,adminorder)
admin.site.register(recipie,adminrecipie)
admin.site.register(totalorder,admintotalorder) 
admin.site.register(sup,adminprofile)
admin.site.register(elections,adminelections)
admin.site.register(candidate,adminCandidate)
admin.site.register(votes,adminvote)




