# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# Note that this schema.rb definition is the authoritative source for your
# database schema. If you need to create the application database on another
# system, you should be using db:schema:load, not running all the migrations
# from scratch. The latter is a flawed and unsustainable approach (the more migrations
# you'll amass, the slower it'll run and the greater likelihood for issues).
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema.define(version: 20170104201226) do

  create_table "exchange_rates", force: :cascade do |t|
    t.string   "base",       limit: 255
    t.string   "currency",   limit: 255
    t.decimal  "rate"
    t.date     "date"
    t.string   "source",     limit: 255
    t.datetime "created_at",             null: false
    t.datetime "updated_at",             null: false
  end

  create_table "fuel_prices", force: :cascade do |t|
    t.decimal  "basic_fuel_price"
    t.decimal  "full_95_coast"
    t.date     "date"
    t.string   "source",                  limit: 255
    t.datetime "created_at",                          null: false
    t.datetime "updated_at",                          null: false
    t.decimal  "exchange_rate"
    t.decimal  "crude_oil"
    t.decimal  "bfp"
    t.decimal  "fuel_tax"
    t.decimal  "customs_excise"
    t.decimal  "equalization_fund_levy"
    t.decimal  "raf"
    t.decimal  "transport_cost"
    t.decimal  "petroleum_products_levy"
    t.decimal  "wholesale_margin"
    t.decimal  "secondary_storage"
    t.decimal  "secondary_distribution"
    t.decimal  "retail_margin"
    t.decimal  "slate_levy"
    t.decimal  "delivery_cost"
    t.decimal  "dsml"
  end

  create_table "oil_prices", force: :cascade do |t|
    t.string   "currency",   limit: 255
    t.decimal  "price"
    t.date     "date"
    t.string   "source",     limit: 255
    t.datetime "created_at",             null: false
    t.datetime "updated_at",             null: false
  end

end
