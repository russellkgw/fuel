STRUCTURE = { 'Date' => { type: 'Date', permitted: [] },
              'Full_95_Coast' => { type: 'Decimal', permitted: [] },
              'Basic_Fuel_Price' => { type: 'Decimal', permitted: [] },
              'Exchange_Rate' => { type: 'Decimal', permitted: [] },
              'Crude_Oil' => { type: 'Decimal', permitted: [] },
              'BFP' => { type: 'Decimal', permitted: [] },
              'Fuel_Tax' => { type: 'Decimal', permitted: [] },
              'Customs_Excise' => { type: 'Decimal', permitted: [] },
              'Equalization_Fund_Levy' => { type: 'Decimal', permitted: [] },
              'RAF' => { type: 'Decimal', permitted: [] },
              'Transport_Cost' => { type: 'Decimal', permitted: [] },
              'Petroleum_Products_Levy' => { type: 'Decimal', permitted: [] },
              'Wholesale_Margin' => { type: 'Decimal', permitted: [] },
              'Secondary_Storage' => { type: 'Decimal', permitted: [] },
              'Secondary_Distribution' => { type: 'Decimal', permitted: [] },
              'Retail_Margin' => { type: 'Decimal', permitted: [] },
              'Slate_Levy' => { type: 'Decimal', permitted: [] },
              'Delivery_Cost' => { type: 'Decimal', permitted: [] },
              'DSML' => { type: 'Decimal', permitted: [] },
              'Source' => { type: 'String', permitted: ['http://www.energy.gov.za/files/petroleum_frame.html'] },
              'Update' => { type: 'String', permitted: ['yes', 'no'] }}

VALID_HEADERS = STRUCTURE.keys

class Data::FuelPriceImport < Data::CsvImport
  def self.import(file)
    fuel_price_import = new
    result, data = fuel_price_import.import(file, VALID_HEADERS)

    if result
      fuel_price_import.insert_data(data)
      result
    else
      result
    end
  end

  def initialize
    super
  end

  def insert_data(data)
    data.each do |item|
      record = FuelPrice.where(date: Date.parse(item[0])).first

      if record.present? && item[20] == 'yes'
        record.update({ full_95_coast: item[1],
                        basic_fuel_price: item[2],
                        exchange_rate: item[3],
                        crude_oil: item[4],
                        bfp: item[5],
                        fuel_tax: item[6],
                        customs_excise: item[7],
                        equalization_fund_levy: item[8],
                        raf: item[9],
                        transport_cost: item[10],
                        petroleum_products_levy: item[11],
                        wholesale_margin: item[12],
                        secondary_storage: item[13],
                        secondary_distribution: item[14],
                        retail_margin: item[15],
                        slate_levy: item[16],
                        delivery_cost: item[17],
                        dsml: item[18],
                        source: item[19]})
      elsif record.blank?
        FuelPrice.create({ date: Date.parse(item[0]),
                           full_95_coast: item[1],
                           basic_fuel_price: item[2],
                           exchange_rate: item[3],
                           crude_oil: item[4],
                           bfp: item[5],
                           fuel_tax: item[6],
                           customs_excise: item[7],
                           equalization_fund_levy: item[8],
                           raf: item[9],
                           transport_cost: item[10],
                           petroleum_products_levy: item[11],
                           wholesale_margin: item[12],
                           secondary_storage: item[13],
                           secondary_distribution: item[14],
                           retail_margin: item[15],
                           slate_levy: item[16],
                           delivery_cost: item[17],
                           dsml: item[18],
                           source: item[19]})
      end
    end
  end
end